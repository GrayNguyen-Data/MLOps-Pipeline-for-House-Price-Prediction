from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
     
    """RegressorMixin: Cho biết đây là mô hình hồi quy -> và nó hỗ trợ các đánh giá score"""
    @abstractmethod
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass

class LinearRegressionStratery(ModelBuildingStrategy):
    def build_train_model(self, X_train, y_train) -> Pipeline:
        
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train không phải là dataframe")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train không phải là dạng series")
        
        logging.info("Khởi tạo mô hình hồi quy tuyến tính  và chuẩn hóa")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)

        logging.info("Hoàn thành việc training model.")
        return pipeline

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Chuyển đổi lựa chọn mô hình khác.")
        self._strategy = strategy
    
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Build và training với mô hình đã chọn.")
        return self._strategy.build_train_model(X_train, y_train)

if __name__ == "__main__":
    pass
