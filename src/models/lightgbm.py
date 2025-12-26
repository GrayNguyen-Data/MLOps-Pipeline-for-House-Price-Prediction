import numpy as np

class _LGBNode:
    __slots__ = ("feature","threshold","left","right","value","gain")
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.gain = None

class LightGBMRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.03, max_leaves=20, min_data_in_leaf=1, lam=1.0, gamma=0.0):
        self.n_estimators = int(n_estimators)     
        self.learning_rate = float(learning_rate)
        self.max_leaves = int(max_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.trees = []
        self.init_pred = 0.0

    def set_params(self, **params):
        for k, v in params.items():
            if k in ("n_estimators", "max_leaves", "min_data_in_leaf"):
                setattr(self, k, int(v))
            elif k in ("learning_rate", "lam", "gamma"):
                setattr(self, k, float(v))
            else:
                setattr(self, k, v)
        return self

    def _calc_grad_hess(self, y, y_pred):
        g = y_pred - y
        h = np.ones_like(g)
        return g, h

    def _best_split_for_node(self, X, g, h):
        G = g.sum(); H = h.sum()
        best_gain = 0.0; best_feat = None; best_thr = None; best_mask = None
        n_features = X.shape[1]
        for feat in range(n_features):
            values = np.unique(X[:, feat])
            if values.shape[0] == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                if left_mask.sum() < self.min_data_in_leaf or left_mask.sum() == X.shape[0]:
                    continue
                G_L = g[left_mask].sum(); H_L = h[left_mask].sum()
                G_R = G - G_L; H_R = H - H_L
                gain = 0.5*(G_L*G_L/(H_L + self.lam) + G_R*G_R/(H_R + self.lam) - G*G/(H + self.lam)) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr
                    best_mask = left_mask

        return best_gain, best_feat, best_thr, best_mask

    def _build_tree_leafwise(self, X, g, h):
        nodes = []
        root = {"idx": np.arange(X.shape[0]), "node": _LGBNode()}
        nodes.append(root)
        G = g.sum(); H = h.sum(); root['node'].value = -G / (H + self.lam)

        leaves = [root]
        while len(leaves) < self.max_leaves:
            best_gain = 0.0; best_leaf = None; best_split = None; best_leaf_idx = None
            for i, leaf in enumerate(leaves):
                idx = leaf['idx']
                if idx.size < 2*self.min_data_in_leaf:
                    continue
                gain, feat, thr, mask = self._best_split_for_node(X[idx], g[idx], h[idx])
                if gain > best_gain and feat is not None:
                    best_gain = gain; best_leaf = leaf; best_split = (feat, thr, mask); best_leaf_idx = i
            if best_leaf is None:
                break
            feat, thr, mask = best_split
            idx = best_leaf['idx']
            left_idx = idx[mask]
            right_idx = idx[~mask]

            node = best_leaf['node']
            node.feature = feat
            node.threshold = thr

            left_node = {"idx": left_idx, "node": _LGBNode()}
            right_node = {"idx": right_idx, "node": _LGBNode()}

            G_L = g[left_idx].sum(); H_L = h[left_idx].sum()
            G_R = g[right_idx].sum(); H_R = h[right_idx].sum()
            node.left = left_node['node']; node.right = right_node['node']
            node.left.value = -G_L / (H_L + self.lam)
            node.right.value = -G_R / (H_R + self.lam)
            node.gain = best_gain
            if best_leaf_idx is not None:
                leaves.pop(best_leaf_idx)
            leaves.append(left_node); leaves.append(right_node)

        return root['node']

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.init_pred = float(y.mean())
        y_pred = np.full(y.shape, self.init_pred)
        self.trees = []
        for m in range(self.n_estimators):
            g, h = self._calc_grad_hess(y, y_pred)
            tree = self._build_tree_leafwise(X, g, h)
            update = np.array([self._predict_row(row, tree) for row in X])
            y_pred = y_pred + self.learning_rate * update
            self.trees.append(tree)
        return self

    def _predict_row(self, x, node):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.full((X.shape[0],), self.init_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * np.array([self._predict_row(row, tree) for row in X])
        return y_pred