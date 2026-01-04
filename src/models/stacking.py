import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import copy
import logging
import time

# ================================
# Cấu hình logging
# ================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model=None, n_folds=5, random_state=None):
        self.base_models = base_models  # Các models cơ sở
        self.meta_model = meta_model    # Model tầng trên
        self.n_folds = n_folds          # Số folds
        self.random_state = random_state
        self.fitted_base_models = []    # Lưu base models đã được train
        
        logger.info("=" * 80)
        logger.info(f"   Khởi tạo StackingRegressor")
        logger.info(f"   Số base models: {len(base_models)}")
        logger.info(f"   Số folds CV:    {n_folds}")
        logger.info("=" * 80)

    def fit(self, X, y):

        start_time = time.time()
        
        # Chuyển sang numpy array để đảm bảo tương thích
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        # Khởi tạo K-Fold Cross Validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Ma trận lưu out-of-fold predictions
        # Shape: (n_samples, n_base_models)
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        logger.info("\n" + "=" * 80)
        logger.info("BƯỚC 1: HUẤN LUYỆN BASE MODELS VỚI K-FOLD CV")
        logger.info("=" * 80)
        
        # ================================
        # Train từng base model
        # ================================
        for i, model in enumerate(self.base_models):
            model_name = type(model).__name__
            logger.info(f"\n Đang train base model {i+1}/{len(self.base_models)}: {model_name}")
            logger.info("-" * 80)
            
            # Lists để lưu metrics của các folds
            fold_train_mse, fold_val_mse = [], []
            fold_train_r2, fold_val_r2 = [], []

            # Loop qua từng fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                # Clone model để tránh ảnh hưởng giữa các folds
                m = copy.deepcopy(model)
                
                # Fit trên fold training data
                m.fit(X[train_idx], y[train_idx])

                # Predict trên cả train và validation
                y_train_pred = m.predict(X[train_idx])
                y_val_pred = m.predict(X[val_idx])
                
                # Lưu OOF predictions cho validation fold
                oof_preds[val_idx, i] = y_val_pred

                # ========================
                # Tính metrics
                # ========================
                train_mse = mean_squared_error(y[train_idx], y_train_pred)
                val_mse = mean_squared_error(y[val_idx], y_val_pred)
                train_r2 = r2_score(y[train_idx], y_train_pred)
                val_r2 = r2_score(y[val_idx], y_val_pred)

                fold_train_mse.append(train_mse)
                fold_val_mse.append(val_mse)
                fold_train_r2.append(train_r2)
                fold_val_r2.append(val_r2)

                # Log kết quả từng fold
                logger.info(
                    f"   [Fold {fold}/{self.n_folds}] "
                    f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
                    f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}"
                )

            # ========================
            # Tính mean metrics qua tất cả folds
            # ========================
            mean_train_mse = np.mean(fold_train_mse)
            mean_val_mse = np.mean(fold_val_mse)
            mean_train_r2 = np.mean(fold_train_r2)
            mean_val_r2 = np.mean(fold_val_r2)
            
            logger.info("-" * 80)
            logger.info(
                f"  {model_name} - Kết quả trung bình qua {self.n_folds} folds:"
            )
            logger.info(
                f"   Train MSE: {mean_train_mse:.4f} | Val MSE: {mean_val_mse:.4f}"
            )
            logger.info(
                f"   Train R²:  {mean_train_r2:.4f} | Val R²:  {mean_val_r2:.4f}"
            )
            
            # Phân tích overfitting
            r2_gap = mean_train_r2 - mean_val_r2
            if r2_gap > 0.1:
                logger.warning(f" Cảnh báo: Gap R² = {r2_gap:.4f} > 0.1 (có dấu hiệu overfitting)")
            elif r2_gap > 0.05:
                logger.info(f"   Gap R² = {r2_gap:.4f} (overfitting nhẹ)")
            else:
                logger.info(f"   Gap R² = {r2_gap:.4f} (tốt)")

        # ================================
        # BƯỚC 2: Train Meta Model
        # ================================
        logger.info("\n" + "=" * 80)
        logger.info("BƯỚC 2: HUẤN LUYỆN META MODEL")
        logger.info("=" * 80)
        logger.info(" OOF predictions đã được tạo cho tất cả base models")
        logger.info(f"   Shape của meta features: {oof_preds.shape}")
        logger.info(" Đang fit meta model trên OOF predictions...")

        # Nếu không có meta_model, dùng Ridge mặc định
        if self.meta_model is None:
            from .ridge import RidgeRegressor
            meta = RidgeRegressor(alpha=1.0)
            logger.info("   Meta model: RidgeRegressor (mặc định) với alpha=1.0")
        else:
            meta = copy.deepcopy(self.meta_model)
            logger.info(f"   Meta model: {type(meta).__name__}")
        
        # Fit meta model
        meta.fit(oof_preds, y)
        self.meta_model_ = meta
        
        # Đánh giá meta model trên OOF predictions
        meta_train_pred = meta.predict(oof_preds)
        meta_train_mse = mean_squared_error(y, meta_train_pred)
        meta_train_r2 = r2_score(y, meta_train_pred)
        
        logger.info("\n  Meta model đã được train")
        logger.info(f"   Meta Train MSE: {meta_train_mse:.4f}")
        logger.info(f"   Meta Train R²:  {meta_train_r2:.4f}")

        # ================================
        # BƯỚC 3: Retrain base models trên toàn bộ data
        # ================================
        logger.info("\n" + "=" * 80)
        logger.info("BƯỚC 3: RETRAIN BASE MODELS TRÊN TOÀN BỘ TRAINING DATA")
        logger.info("=" * 80)
        
        self.fitted_base_models = []
        
        for i, m in enumerate(self.base_models):
            model_name = type(m).__name__
            logger.info(f" Đang retrain {model_name} ({i+1}/{len(self.base_models)})...")
            
            cloned = copy.deepcopy(m)
            cloned.fit(X, y)
            self.fitted_base_models.append(cloned)
            
            # Đánh giá trên toàn bộ training set
            y_pred_full = cloned.predict(X)
            full_mse = mean_squared_error(y, y_pred_full)
            full_r2 = r2_score(y, y_pred_full)
            
            logger.info(
                f"   {model_name} - Full Train MSE: {full_mse:.4f} | R²: {full_r2:.4f}"
            )

        # ================================
        # Hoàn tất
        # ================================
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info(" STACKING REGRESSOR HUẤN LUYỆN HOÀN TẤT")
        logger.info("=" * 80)
        logger.info(f" Tổng thời gian: {elapsed_time:.2f}s")
        logger.info(f" Số base models: {len(self.fitted_base_models)}")
        logger.info(f" Meta model: {type(self.meta_model_).__name__}")
        logger.info("=" * 80 + "\n")
        
        return self

    def predict(self, X):
        logger.info(" Đang dự đoán với StackingRegressor...")
        
        X = np.asarray(X)
        
        # Kiểm tra đã fit chưa
        if not self.fitted_base_models:
            raise RuntimeError(" Model chưa được train. Gọi fit() trước khi predict().")
        
        # Bước 1: Lấy predictions từ tất cả base models
        logger.info(f" Đang lấy predictions từ {len(self.fitted_base_models)} base models...")
        base_predictions = [m.predict(X) for m in self.fitted_base_models]
        
        # Bước 2: Stack predictions thành meta features
        # Shape: (n_samples, n_base_models)
        meta_features = np.column_stack(base_predictions)
        logger.info(f"  Meta features shape: {meta_features.shape}")
        
        # Bước 3: Meta model dự đoán cuối cùng
        logger.info("  Meta model đang kết hợp predictions...")
        preds = self.meta_model_.predict(meta_features)
        
        logger.info(f" Hoàn thành dự đoán cho {len(preds)} mẫu\n")
        
        return preds