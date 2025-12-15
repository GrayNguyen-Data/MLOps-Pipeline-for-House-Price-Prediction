from  abc import ABC, abstractmethod

import pandas as pd
import numpy as np 
import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

logging.basicConfig(level = logging.INFO, format ="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        self._features = features
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật lấy log cho features")

        df_transformed = df.copy()
        for feature in self._features:
            df_transformed[feature] = np.log1p(df[feature])
        
        logging.info("Đã hoàn thành việc lấy log feature.")

        return df_transformed

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        self._features = features
        self.scaler = StandardScaler()
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật StandardScaling cho features.")
        df_transformed = df.copy()
        df_transformed[self._features] = self.scaler.fit_transform(df[self._features])
        logging.info("Đã hoàn thành việc scale bằng phương pháp StandardScaling cho feature.")
        return df_transformed

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list, feature_range=(0,1)):
        self._features = features
        self.scaler =MinMaxScaler(feature_range=feature_range)
    
    def transformation(self, df:pd.DataFrame) ->pd.DataFrame:
        logging.info("Áp dụng kĩ thuật MinMaxScaling cho các features.")
        df_transformed = df.copy()
        df_transformed[self._features] = self.scaler.fit_transform(df[self._features])
        logging.info("Đã hoàn thành việc sacle bằng phương pháp MinMaxScaling cho feature.")
        return df_transformed

class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """ 
            - spares = false -> trả về một mảng numpy
            - drop = 'first' -> xóa đi cột đầu tiên, nhằm mục đích tránh overfiting bởi vì cột đầu tiên = 1 -(tất cả các cột còn lại) -> có mối quan hệ mật thiết.
        """
        self._features = features
        self.encoder = OneHotEncoder(sparse=False, drop = 'first')
    
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Áp dụng kĩ thuật OneHotEncoding cho các features.")
        df_transformed = df.copy()
        encoder_df = pd.DataFrame(self.encoder.fit_transform(df[self._features]), columns = self.encoder.get_feature_names_out(self._features))
        df_transformed = df_transformed.drop(columns=self._features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoder_df], axis = 1)
        logging.info("Đã hoàn thành việc scale bằng phương pháp OneHotEncoding cho features.")
        return df_transformed
    
class FeatureEngineer:
    def __init__(self, stratery: FeatureEngineeringStrategy):
        self._stratery = stratery
    
    def set_stratery(self, stratery: FeatureEngineeringStrategy):
        self._stratery = stratery
    
    def apply_Transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Hoàn thành việc sử dụng các kĩ thuật scale cho feature.")
        return self._stratery.transformation(df)

if __name__ == "__main__":
    pass