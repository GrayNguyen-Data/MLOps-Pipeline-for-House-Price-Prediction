from zenml import step, Model
from zenml.steps import get_step_context
from typing import Annotated, Dict
import logging
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# Cấu hình logging
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ================================
# Metadata mô hình cho ZenML
# ================================
model_object = Model(
    name="prices_predictor",
    description="Mô hình dự đoán giá nhà khu vực TP. Hồ Chí Minh",
)

@step(
    enable_cache=False,
    model=model_object,
)
def model_evaluator_step(
    trained_model: Annotated[Pipeline, "trained_model"],
    X_test: Annotated[pd.DataFrame, "X_test"],
    y_test: Annotated[pd.DataFrame, "y_test"],
) -> Annotated[Dict[str, float], "evaluation_metrics"]:

    logging.info("=" * 80)
    logging.info("BẮT ĐẦU BƯỚC ĐÁNH GIÁ MÔ HÌNH")
    logging.info("=" * 80)

    # ================================
    # 1. Chuyển y_test thành Series nếu cần
    # ================================
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
        logging.info(" Đã chuyển y_test từ DataFrame sang Series")

    # ================================
    # 2. Dự đoán trên tập test
    # ================================
    logging.info(" Đang thực hiện dự đoán trên tập test...")
    y_pred = trained_model.predict(X_test)
    logging.info(f" Hoàn thành dự đoán cho {len(y_pred)} mẫu")

    # ================================
    # 3. Tính toán các metrics
    # ================================
    logging.info(" Đang tính toán các chỉ số đánh giá...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info("\n" + "=" * 80)
    logging.info(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    logging.info("=" * 80)
    logging.info(f"   MSE (Mean Squared Error): {mse:.4f}")
    logging.info(f"   R² Score:                 {r2:.4f}")
    logging.info(f"   Số features:              {X_test.shape[1]}")
    logging.info(f"   Số mẫu test:              {X_test.shape[0]}")
    logging.info("=" * 80)

    # ================================
    # 4. Lưu metadata vào Step Context (hiển thị trên Dashboard)
    # ================================
    logging.info("\n Đang lưu metadata vào output artifact...")
    step_context = get_step_context()

    # Lưu metadata cho artifact output - hiển thị trên ZenML Dashboard
    step_context.add_output_metadata(
        output_name="evaluation_metrics",
        metadata={
            "mse": float(mse),
            "r2": float(r2),
            "num_features": int(X_test.shape[1]),
            "num_test_samples": int(X_test.shape[0]),
        },
    )

    logging.info(" Đã lưu metrics vào output artifact metadata")

    # ================================
    # 5. Lưu metadata vào Model Version
    # ================================
    logging.info(" Đang lưu metadata vào model version...")
    model_version = step_context.model

    model_version.log_metadata(
        {
            "mse": float(mse),
            "r2": float(r2),
            "num_features": int(X_test.shape[1]),
            "num_test_samples": int(X_test.shape[0]),
            "r2_threshold": 0.85,  # Ngưỡng để promote lên production
        }
    )

    logging.info(" Đã lưu metrics vào model version metadata")

    # ================================
    # 6. Logic Promotion (Đưa model lên Production)
    # ================================
    logging.info("\n  Đang kiểm tra điều kiện promotion...")
    logging.info(f"   Ngưỡng R² tối thiểu: 0.85")
    logging.info(f"   R² hiện tại:         {r2:.4f}")

    if r2 >= 0.85:
        # Model đạt yêu cầu → Promote lên Production
        model_version.set_stage("production", force=True)
        model_version.log_metadata({"promoted_to_production": True})

        logging.info("\n" + "=" * 80)
        logging.info(f" THÀNH CÔNG! Model version {model_version.version} đã được PROMOTE lên PRODUCTION")
        logging.info("=" * 80)
    else:
        # Model chưa đạt yêu cầu → Không promote
        model_version.log_metadata({"promoted_to_production": False})
        
        logging.warning("\n" + "=" * 80)
        logging.warning(f"  Model KHÔNG được promote (R² = {r2:.4f} < 0.85)")
        logging.warning("   Gợi ý: Cần cải thiện mô hình hoặc điều chỉnh hyperparameters")
        logging.warning("=" * 80)

    # ================================
    # 7. Kết thúc và trả về metrics
    # ================================
    logging.info("\n" + "=" * 80)
    logging.info(" KẾT THÚC BƯỚC ĐÁNH GIÁ MÔ HÌNH")
    logging.info("=" * 80)
    logging.info(" Kiểm tra ZenML Dashboard để xem chi tiết metrics")
    logging.info("=" * 80 + "\n")

    # Trả về dictionary chứa metrics
    return {
        "mse": float(mse),
        "r2": float(r2),
    }