# from typing import Annotated
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from zenml import step, Model
# import logging

# # ================================
# # Import models & tuner
# # ================================
# from models.stacking import StackingRegressor
# from models.ridge import RidgeRegressor
# from models.xgboost import XGBoostRegressor
# from models.lightgbm import LightGBMRegressor
# from models.random_forest import RandomForestRegressor
# from models.linear import LinearRegressor
# from models.tuning import grid_search_with_metrics

# # ================================
# # 1️⃣ Logger setup (UTF-8 safe)
# # ================================
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_FILE = os.path.join(LOG_DIR, "model_building.log")

# logger = logging.getLogger("stacking_logger")
# logger.setLevel(logging.INFO)

# formatter = logging.Formatter(
#     "%(asctime)s - %(levelname)s - %(message)s"
# )

# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)

# # ================================
# # 2️⃣ ZenML Model metadata
# # ================================
# model = Model(
#     name="prices_predictor",
#     version=None,
#     license="Apache 2.0",
#     description="Mô hình dự đoán giá nhà",
# )

# # ================================
# # 3️⃣ Helper: Train vs Val per fold
# # ================================
# def plot_train_vs_val_per_fold(df_metrics: pd.DataFrame, model_name: str):
#     """
#     Vẽ Train vs Validation R² cho TỪNG FOLD
#     -> dùng để phát hiện overfitting theo hành vi, KHÔNG dùng mean
#     """
#     os.makedirs("figures", exist_ok=True)
#     sns.set(style="whitegrid")

#     df_plot = df_metrics.copy()
#     df_plot["fold"] = range(1, len(df_plot) + 1)

#     df_plot = df_plot.melt(
#         id_vars="fold",
#         value_vars=["train_r2", "val_r2"],
#         var_name="Dataset",
#         value_name="R2",
#     )

#     plt.figure(figsize=(10, 5))
#     sns.barplot(
#         data=df_plot,
#         x="fold",
#         y="R2",
#         hue="Dataset",
#     )

#     plt.title(f"{model_name} - Train vs Validation R² per fold")
#     plt.xlabel("Fold")
#     plt.ylabel("R² Score")
#     plt.ylim(0, 1.05)

#     plt.tight_layout()
#     plt.savefig(
#         f"figures/{model_name}_train_vs_val_per_fold.png",
#         dpi=300,
#     )
#     plt.close()

# # ================================
# # 4️⃣ ZenML Step chính
# # ================================
# @step(enable_cache=True, model=model)
# def model_building_step(
#     X_train: Annotated[pd.DataFrame, "X_train"],
#     y_train: Annotated[pd.DataFrame, "y_train"],
# ) -> Annotated[Pipeline, "sklearn_pipeline"]:
#     """
#     Train base models với GridSearch
#     + vẽ Train vs Validation R² theo TỪNG FOLD
#     """
#     logger.info("=" * 80)
#     logger.info("MODEL BUILDING STEP - auto tuning base models (metric = R2)")
#     logger.info("=" * 80)

#     y_train_series = y_train.iloc[:, 0]

#     # ============================
#     # Hyperparameter spaces
#     # ============================
#     param_spaces = {
#         "Ridge": {"alpha": [0.1, 1.0, 10.0]},
#         "XGBoost": {
#             "n_estimators": [50, 100],
#             "learning_rate": [0.03, 0.05],
#             "max_depth": [3, 5],
#         },
#         "LightGBM": {
#             "n_estimators": [50, 100],
#             "learning_rate": [0.03, 0.05],
#             "max_leaves": [15, 20],
#         },
#         "RandomForest": {
#             "n_estimators": [50, 100],
#             "max_depth": [5, 10],
#         },
#     }

#     base_models = []
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)

#     logger.info("Bắt đầu Grid Search cho từng base model...\n")

#     for name, (ModelClass, params) in zip(
#         param_spaces.keys(),
#         zip(
#             [
#                 RidgeRegressor,
#                 XGBoostRegressor,
#                 LightGBMRegressor,
#                 RandomForestRegressor,
#             ],
#             param_spaces.values(),
#         ),
#     ):
#         logger.info(f"{name}: bắt đầu tuning (metric = R2)")

#         best_model, best_params, df_metrics = grid_search_with_metrics(
#             ModelClass(),
#             params,
#             X_train.values,
#             y_train_series.values,
#             model_name=name,
#         )

#         base_models.append(best_model)

#         # ========================
#         # Save metrics
#         # ========================
#         csv_path = os.path.join(
#             results_dir, f"{name}_r2_metrics.csv"
#         )
#         df_metrics.to_csv(csv_path, index=False)

#         logger.info(f"{name}: metrics lưu tại {csv_path}")

#         # ========================
#         # Visualization (per fold)
#         # ========================
#         plot_train_vs_val_per_fold(df_metrics, name)
#         logger.info(
#             f"{name}: đã vẽ Train vs Validation R2 cho từng fold"
#         )

#         # ========================
#         # Overfitting diagnosis
#         # ========================
#         overfit_folds = (
#             df_metrics["train_r2"] - df_metrics["val_r2"] > 0.1
#         ).sum() 

#         logger.info(
#             f"{name}: {overfit_folds}/{len(df_metrics)} folds có dấu hiệu overfitting"
#         )

#     # ================================
#     # 5️⃣ Stacking layer
#     # ================================
#     logger.info("\nHuấn luyện StackingRegressor...")

#     stack_model = StackingRegressor(
#         base_models=base_models,
#         meta_model=LinearRegressor(),
#         n_folds=5,
#     )

#     pipeline = Pipeline(
#         [
#             ("scaler", StandardScaler()),
#             ("model", stack_model),
#         ]
#     )

#     logger.info("Fitting pipeline...")
#     pipeline.fit(X_train, y_train_series)

#     logger.info("Training hoàn tất.")
#     logger.info("=" * 80)

#     return pipeline

from typing import Annotated
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from zenml import step, Model
import logging
import numpy as np

# ================================
# Import các models & tuner
# ================================
from models.stacking import StackingRegressor
from models.ridge import RidgeRegressor
from models.xgboost import XGBoostRegressor
from models.lightgbm import LightGBMRegressor
from models.random_forest import RandomForestRegressor
from models.linear import LinearRegressor
from models.tuning import grid_search_with_metrics

# ================================
# 1️⃣ Cấu hình Logger (hỗ trợ UTF-8)
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
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
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
# 3️⃣ Các hàm Visualization
# ================================
def plot_comprehensive_analysis(df_metrics: pd.DataFrame, model_name: str):
    os.makedirs("figures", exist_ok=True)
    
    # Thiết lập style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Tạo figure với 4 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Tính toán overfitting gap
    df_metrics['overfit_gap'] = df_metrics['train_r2'] - df_metrics['val_r2']
    df_metrics['fold'] = range(1, len(df_metrics) + 1)
    
    # ============ Biểu đồ 1: Train vs Val per fold ============
    ax1 = fig.add_subplot(gs[0, :])
    x = df_metrics['fold']
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_metrics['train_r2'], width, 
                    label='Train R²', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, df_metrics['val_r2'], width,
                    label='Validation R²', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Thêm nhãn giá trị
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Điểm R²', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - Train vs Validation R² theo từng Fold', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ============ Biểu đồ 2: Overfitting Gap ============
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#FF6B6B' if gap > 0.1 else '#4ECDC4' if gap > 0.05 else '#95E1D3' 
              for gap in df_metrics['overfit_gap']]
    
    bars = ax2.bar(df_metrics['fold'], df_metrics['overfit_gap'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Ngưỡng: 0.05')
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Rủi ro cao: 0.10')
    
    ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Khoảng cách Overfitting (Train R² - Val R²)', fontsize=11, fontweight='bold')
    ax2.set_title('Phân tích Overfitting theo từng Fold', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(df_metrics['fold'])
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ============ Biểu đồ 3: Mean Comparison ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    mean_train = df_metrics['train_r2'].mean()
    mean_val = df_metrics['val_r2'].mean()
    std_train = df_metrics['train_r2'].std()
    std_val = df_metrics['val_r2'].std()
    
    metrics = ['Train R²', 'Val R²']
    means = [mean_train, mean_val]
    stds = [std_train, std_val]
    colors_bar = ['#2E86AB', '#A23B72']
    
    bars = ax3.bar(metrics, means, yerr=stds, capsize=10, 
                   color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Thêm nhãn giá trị
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax3.text(bar.get_x() + bar.get_width()/2., mean + std,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Điểm R²', fontsize=12, fontweight='bold')
    ax3.set_title('Hiệu suất Trung bình với Độ lệch chuẩn', fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============ Biểu đồ 4: Distribution Box Plot ============
    ax4 = fig.add_subplot(gs[2, :])
    
    data_to_plot = [df_metrics['train_r2'], df_metrics['val_r2']]
    bp = ax4.boxplot(data_to_plot, labels=['Train R²', 'Validation R²'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Tô màu các hộp
    colors_box = ['#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Thêm scatter points
    for i, data in enumerate(data_to_plot, 1):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        ax4.scatter(x, y, alpha=0.6, s=50, color='black', edgecolor='white', linewidth=0.5)
    
    ax4.set_ylabel('Điểm R²', fontsize=12, fontweight='bold')
    ax4.set_title('Phân phối Điểm số qua các Fold', fontsize=13, fontweight='bold', pad=15)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Lưu hình
    plt.savefig(f"figures/{model_name}_comprehensive_analysis.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f" Đã lưu biểu đồ phân tích tổng hợp cho {model_name}")


def diagnose_overfitting(df_metrics: pd.DataFrame, model_name: str):
    """
    Phân tích chi tiết hiện tượng overfitting và đưa ra kết luận
    """
    df_metrics['overfit_gap'] = df_metrics['train_r2'] - df_metrics['val_r2']
    
    mean_train = df_metrics['train_r2'].mean()
    mean_val = df_metrics['val_r2'].mean()
    mean_gap = df_metrics['overfit_gap'].mean()
    max_gap = df_metrics['overfit_gap'].max()
    min_gap = df_metrics['overfit_gap'].min()
    std_gap = df_metrics['overfit_gap'].std()
    
    # Đếm số fold có vấn đề
    high_overfit_folds = (df_metrics['overfit_gap'] > 0.1).sum()
    moderate_overfit_folds = ((df_metrics['overfit_gap'] > 0.05) & 
                              (df_metrics['overfit_gap'] <= 0.1)).sum()
    good_folds = (df_metrics['overfit_gap'] <= 0.05).sum()
    
    logger.info(f"\n{'='*80}")
    logger.info(f" CHẨN ĐOÁN OVERFITTING - {model_name}")
    logger.info(f"{'='*80}")
    logger.info(f" R² Trung bình (Train):      {mean_train:.4f}")
    logger.info(f" R² Trung bình (Validation): {mean_val:.4f}")
    logger.info(f" Gap Trung bình:             {mean_gap:.4f}")
    logger.info(f" Khoảng Gap:                 [{min_gap:.4f}, {max_gap:.4f}]")
    logger.info(f" Độ lệch chuẩn Gap:          {std_gap:.4f}")
    logger.info(f"\nPhân tích theo Fold:")
    logger.info(f"  Fold tốt (gap ≤ 0.05):              {good_folds}/{len(df_metrics)}")
    logger.info(f"  Overfitting vừa phải (0.05-0.10):  {moderate_overfit_folds}/{len(df_metrics)}")
    logger.info(f"  Overfitting cao (gap > 0.10):       {high_overfit_folds}/{len(df_metrics)}")
    
    # Đưa ra kết luận
    logger.info(f"\nKẾT LUẬN:")
    
    if mean_gap <= 0.03 and high_overfit_folds == 0:
        conclusion = " XUẤT SẮC - Không có dấu hiệu overfitting. Mô hình generalize tốt."
        severity = "NONE"
    elif mean_gap <= 0.05 and high_overfit_folds == 0:
        conclusion = " TỐT - Overfitting ở mức chấp nhận được. Mô hình ổn định."
        severity = "LOW"
    elif mean_gap <= 0.08 or (mean_gap <= 0.1 and high_overfit_folds <= 1):
        conclusion = " VỪA PHẢI - Có dấu hiệu overfitting nhẹ. Nên xem xét regularization."
        severity = "MODERATE"
    else:
        conclusion = " RỦI RO CAO - Overfitting nghiêm trọng. Cần điều chỉnh mô hình hoặc tăng regularization."
        severity = "HIGH"
    
    logger.info(f"   {conclusion}")
    logger.info(f"{'='*80}\n")
    
    return {
        'model_name': model_name,
        'mean_train_r2': mean_train,
        'mean_val_r2': mean_val,
        'mean_gap': mean_gap,
        'max_gap': max_gap,
        'std_gap': std_gap,
        'high_overfit_folds': high_overfit_folds,
        'moderate_overfit_folds': moderate_overfit_folds,
        'good_folds': good_folds,
        'severity': severity,
        'conclusion': conclusion
    }


def plot_final_model_performance(pipeline, X_train, y_train, model_name="Stacking Model"):
    """
    Vẽ hiệu suất của model cuối cùng trên toàn bộ tập training
    """
    os.makedirs("figures", exist_ok=True)
    
    # Dự đoán trên training data
    y_pred = pipeline.predict(X_train)
    
    # Tính các metrics
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    
    # Tạo figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(f'{model_name} - Hiệu suất Cuối cùng trên Toàn bộ Tập Training', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ============ Plot 1: Actual vs Predicted ============
    ax1 = axes[0]
    ax1.scatter(y_train, y_pred, alpha=0.5, s=30, color='#2E86AB', edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_train.min(), y_pred.min())
    max_val = max(y_train.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Dự đoán Hoàn hảo')
    
    ax1.set_xlabel('Giá trị Thực tế', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Giá trị Dự đoán', fontsize=12, fontweight='bold')
    ax1.set_title('Thực tế vs Dự đoán', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Thêm text box với metrics
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # ============ Plot 2: Residuals ============
    ax2 = axes[1]
    residuals = y_train - y_pred
    
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30, color='#A23B72', edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual = 0')
    
    ax2.set_xlabel('Giá trị Dự đoán', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Thực tế - Dự đoán)', fontsize=12, fontweight='bold')
    ax2.set_title('Biểu đồ Residual', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Thêm text box với residual stats
    textstr = f'Trung bình = {residuals.mean():.4f}\nĐộ lệch chuẩn = {residuals.std():.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_final_performance.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   Đã lưu biểu đồ performance của model cuối cùng")
    logger.info(f"   R² Score: {r2:.4f}")
    logger.info(f"   RMSE: {rmse:.4f}")
    logger.info(f"   MAE: {mae:.4f}")
    
    return r2, rmse, mae


def create_summary_comparison(all_diagnostics: list):
    """
    Tạo bảng so sánh tổng hợp tất cả các mô hình
    """
    os.makedirs("figures", exist_ok=True)
    
    df_summary = pd.DataFrame(all_diagnostics)
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Chuẩn bị dữ liệu cho bảng
    table_data = []
    table_data.append(['Mô hình', 'Train R²', 'Val R²', 'Gap', 'Fold Rủi ro', 'Mức độ', 'Trạng thái'])
    
    for _, row in df_summary.iterrows():
        status_emoji = {
            'NONE': 'None',
            'LOW': 'Low',
            'MODERATE': 'Moderate',
            'HIGH': 'High'
        }.get(row['severity'], '?')
        
        table_data.append([
            row['model_name'],
            f"{row['mean_train_r2']:.4f}",
            f"{row['mean_val_r2']:.4f}",
            f"{row['mean_gap']:.4f}",
            f"{row['high_overfit_folds']}",
            row['severity'],
            status_emoji
        ])
    
    # Tạo bảng
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.15, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Định dạng header
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Định dạng data rows
    for i in range(1, len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
    
    plt.title('So sánh Mô hình - Tổng hợp Phân tích Overfitting', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('figures/model_comparison_summary.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Lưu CSV
    df_summary.to_csv('results/model_comparison_summary.csv', index=False)
    
    logger.info(" Đã lưu bảng so sánh tổng hợp các models")


# ================================
# 4️⃣ ZenML Step chính
# ================================
@step(enable_cache=True, model=model)
def model_building_step(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.DataFrame, "y_train"],
) -> Annotated[Pipeline, "sklearn_pipeline"]:
    """
    Huấn luyện base models với GridSearch
    + Phân tích overfitting toàn diện
    + Visualization đẹp và chi tiết
    """
    logger.info("=" * 80)
    logger.info("BƯỚC HUẤN LUYỆN MÔ HÌNH - Tự động tuning với phân tích chi tiết")
    logger.info("=" * 80)

    y_train_series = y_train.iloc[:, 0]

    # ============================
    # Không gian Hyperparameter
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
    all_diagnostics = []
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(" Bắt đầu Grid Search cho từng base model...\n")

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
        logger.info(f"\n{'='*80}")
        logger.info(f" {name}: Bắt đầu tuning (metric = R2)")
        logger.info(f"{'='*80}")

        best_model, best_params, df_metrics = grid_search_with_metrics(
            ModelClass(),
            params,
            X_train.values,
            y_train_series.values,
            model_name=name,
        )

        base_models.append(best_model)

        # Lưu metrics
        csv_path = os.path.join(results_dir, f"{name}_r2_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        logger.info(f" Đã lưu metrics: {csv_path}")

        # Vẽ biểu đồ phân tích tổng hợp
        plot_comprehensive_analysis(df_metrics, name)

        # Chẩn đoán overfitting
        diagnosis = diagnose_overfitting(df_metrics, name)
        all_diagnostics.append(diagnosis)

    # ================================
    # Tạo bảng so sánh
    # ================================
    logger.info("\n" + "="*80)
    logger.info(" Đang tạo bảng so sánh tổng hợp các models...")
    logger.info("="*80)
    create_summary_comparison(all_diagnostics)

    # ================================
    # 5️⃣ Stacking layer
    # ================================
    logger.info("\n" + "="*80)
    logger.info(" Đang huấn luyện StackingRegressor (Meta Model)...")
    logger.info("="*80)

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

    logger.info(" Đang fit pipeline...")
    pipeline.fit(X_train, y_train_series)

    # ================================
    # Đánh giá model cuối cùng
    # ================================
    logger.info("\n" + "="*80)
    logger.info(" Đang đánh giá model cuối cùng trên toàn bộ tập training...")
    logger.info("="*80)
    
    plot_final_model_performance(pipeline, X_train, y_train_series, "Stacking_Model")

    logger.info("\n" + "="*80)
    logger.info(" HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG")
    logger.info("="*80)
    logger.info(" Kiểm tra thư mục 'figures/' để xem tất cả biểu đồ")
    logger.info(" Kiểm tra thư mục 'results/' để xem metrics và bảng tổng hợp")
    logger.info("="*80 + "\n")

    return pipeline