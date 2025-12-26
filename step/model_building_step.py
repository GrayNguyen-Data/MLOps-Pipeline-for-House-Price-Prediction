from typing import Annotated
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml import step, Model
import logging

# ================================
# Import models & tuner
# ================================
from models.stacking import StackingRegressor
from models.ridge import RidgeRegressor
from models.xgboost import XGBoostRegressor
from models.lightgbm import LightGBMRegressor
from models.random_forest import RandomForestRegressor
from models.linear import LinearRegressor
from models.tuning import grid_search_with_metrics

# ================================
# 1️⃣ Logger setup (UTF-8 safe)
# ================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "model_building.log")

logger = logging.getLogger("stacking_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# ================================
# 2️⃣ ZenML Model metadata
# ================================
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Mô hình dự đoán giá nhà",
)

# ================================
# 3️⃣ Helper: Train vs Val per fold
# ================================
def plot_train_vs_val_per_fold(df_metrics: pd.DataFrame, model_name: str):
    """
    Vẽ Train vs Validation R² cho TỪNG FOLD
    -> dùng để phát hiện overfitting theo hành vi, KHÔNG dùng mean
    """
    os.makedirs("figures", exist_ok=True)
    sns.set(style="whitegrid")

    df_plot = df_metrics.copy()
    df_plot["fold"] = range(1, len(df_plot) + 1)

    df_plot = df_plot.melt(
        id_vars="fold",
        value_vars=["train_r2", "val_r2"],
        var_name="Dataset",
        value_name="R2",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_plot,
        x="fold",
        y="R2",
        hue="Dataset",
    )

    plt.title(f"{model_name} - Train vs Validation R² per fold")
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(
        f"figures/{model_name}_train_vs_val_per_fold.png",
        dpi=300,
    )
    plt.close()

# ================================
# 4️⃣ ZenML Step chính
# ================================
@step(enable_cache=True, model=model)
def model_building_step(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.DataFrame, "y_train"],
) -> Annotated[Pipeline, "sklearn_pipeline"]:
    """
    Train base models với GridSearch
    + vẽ Train vs Validation R² theo TỪNG FOLD
    """
    logger.info("=" * 80)
    logger.info("MODEL BUILDING STEP - auto tuning base models (metric = R2)")
    logger.info("=" * 80)

    y_train_series = y_train.iloc[:, 0]

    # ============================
    # Hyperparameter spaces
    # ============================
    param_spaces = {
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "XGBoost": {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 5],
        },
        "LightGBM": {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.05],
            "max_leaves": [15, 20],
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
        },
    }

    base_models = []
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Bắt đầu Grid Search cho từng base model...\n")

    for name, (ModelClass, params) in zip(
        param_spaces.keys(),
        zip(
            [
                RidgeRegressor,
                XGBoostRegressor,
                LightGBMRegressor,
                RandomForestRegressor,
            ],
            param_spaces.values(),
        ),
    ):
        logger.info(f"{name}: bắt đầu tuning (metric = R2)")

        best_model, best_params, df_metrics = grid_search_with_metrics(
            ModelClass(),
            params,
            X_train.values,
            y_train_series.values,
            model_name=name,
        )

        base_models.append(best_model)

        # ========================
        # Save metrics
        # ========================
        csv_path = os.path.join(
            results_dir, f"{name}_r2_metrics.csv"
        )
        df_metrics.to_csv(csv_path, index=False)

        logger.info(f"{name}: metrics lưu tại {csv_path}")

        # ========================
        # Visualization (per fold)
        # ========================
        plot_train_vs_val_per_fold(df_metrics, name)
        logger.info(
            f"{name}: đã vẽ Train vs Validation R2 cho từng fold"
        )

        # ========================
        # Overfitting diagnosis
        # ========================
        overfit_folds = (
            df_metrics["train_r2"] - df_metrics["val_r2"] > 0.1
        ).sum() 

        logger.info(
            f"{name}: {overfit_folds}/{len(df_metrics)} folds có dấu hiệu overfitting"
        )

    # ================================
    # 5️⃣ Stacking layer
    # ================================
    logger.info("\nHuấn luyện StackingRegressor...")

    stack_model = StackingRegressor(
        base_models=base_models,
        meta_model=LinearRegressor(),
        n_folds=5,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", stack_model),
        ]
    )

    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train_series)

    logger.info("Training hoàn tất.")
    logger.info("=" * 80)

    return pipeline
