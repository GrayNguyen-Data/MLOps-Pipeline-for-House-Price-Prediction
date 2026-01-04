# File: training_pipeline.py (Phiên bản TỐI ƯU/TỰ ĐỘNG)
import sys
from pathlib import Path

# Ensure project `src` directory is on sys.path so imports like `from models...` work
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from step.data_ingestion_step import data_ingestion_step
from step.data_splitter_step import data_splitter_step
from step.feature_engineering_step import feature_engineering_step
from step.handle_missing_value_step import handle_missing_values_step
from step.model_building_step import model_building_step
from step.evaluator_model_step import model_evaluator_step
from step.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline
from typing import Annotated, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@pipeline(
    model=Model(name="prices_predictor"),
    enable_cache=True
)
def ml_pipeline() -> Tuple[Annotated[Pipeline, "trained_model_pipeline"], Annotated[dict, "evaluation_metrics"]]:
    """Define an end-to-end machine learning pipeline."""

    logging.info("--- BẮT ĐẦU ML PIPELINE ---")
    target_column = "SalePrice"
    
    # 1. Data Ingestion
    raw_data: Annotated[pd.DataFrame, ArtifactConfig("raw_data")] = data_ingestion_step(
        file_path="D:\\Project_Portfolio\\HOUSE-PRICE-MLOPS\\data\\storage.zip"
    )

    # 2. Handling Missing Values - NUMERIC COLUMNS (Điền mean cho các cột số)
    filled_numeric_data: Annotated[pd.DataFrame, ArtifactConfig("filled_numeric_data")] = handle_missing_values_step(
        df=raw_data, 
        strategy="mean"
    )

    # 3. Handling Missing Values - CATEGORICAL COLUMNS (BẮT BUỘC TRƯỚC OHE)
    # Điền giá trị thiếu bằng chuỗi "Missing" để OHE không gặp lỗi NaN.
    filled_data_final: Annotated[pd.DataFrame, ArtifactConfig("filled_data_final")] = handle_missing_values_step(
        df=filled_numeric_data, 
        strategy="constant",
        fill_value="Missing" 
    )

    # 4. FEATURE ENGINEERING: ONE-HOT ENCODING (Tự động chọn cột object)
    # features=None sẽ kích hoạt logic tự động tìm cột object đã sửa trong src/feature_engineering.py.
    encoded_data: Annotated[pd.DataFrame, ArtifactConfig("encoded_data")] = feature_engineering_step(
        df=filled_data_final, 
        strategy="onehot_encoding", 
        features=None 
    )

    # 5. Feature Engineering: LOG TRANSFORMATION (Áp dụng cho cột số)
    engineered_data: Annotated[pd.DataFrame, ArtifactConfig("engineered_data")] = feature_engineering_step(
        df=encoded_data, 
        strategy="log",
        features=["Gr Liv Area", target_column] # Vẫn cần chỉ định thủ công các cột cần Log Transform
    )
    
    # 6. Outlier Detection Step
    clean_data: Annotated[pd.DataFrame, ArtifactConfig("clean_data")] = outlier_detection_step(
        df=engineered_data
    )

    # 7. Data Splitting Step
    X_train, y_train, X_test, y_test = data_splitter_step(
        df=clean_data,
        target_column=target_column
    )

    # 8. Model Building Step
    trained_model: Annotated[Pipeline, ArtifactConfig("sklearn_pipeline")] = model_building_step(
        X_train=X_train, 
        y_train=y_train
    )

    # 9. Model Evaluation Step
    evaluation_metrics: Annotated[dict, ArtifactConfig("evaluation_metrics")] = model_evaluator_step(
        trained_model=trained_model,
        X_test=X_test,
        y_test=y_test
    )
    
    logging.info("--- HOÀN TẤT ML PIPELINE ---")

    return trained_model, evaluation_metrics