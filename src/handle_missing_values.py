from abc import ABC, abstractmethod
import pandas as pd
import logging

"""Thiết lập thông báo lỗi"""
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValueHandlingStrategy(ABC): 

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
class DropMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis = 0, thresh =None):
        """ 
        - axis = 0 -> xóa hàng bị thiếu
        - axis = 1 -> xóa cột cột bị thiếu
        - thresh: int -> số lượng giá trị không bị NA tối thiểu để row/column được giữ lại.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Đã xóa các giá trị bị thiếu với axis={self.axis} và thresh={self.thresh}")
        df_clean = df.dropna(axis = self.axis, thresh =self.thresh)
        logging.info("Các giá trị bị thiếu đã được xóa.")
        return df_clean
    
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method ="mean", fill_value="None"):
        
        self.method =method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:

        logging.info(f"Điền các giá trị thiếu với method={self.method}")
        df_cleand = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleand.select_dtypes(include=" number").columns
            df_cleand[numeric_columns] = df_cleand[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == "median":
            numeric_columns = df_cleand.select_dtypes(include=" number").columns
            df_cleand[numeric_columns] = df_cleand[numeric_columns].fillna(df[numeric_columns].median())
        
        elif self.method == "mode":  #mode: tần xuất xuất hiện
            numeric_columns = df_cleand.select_dtypes(include=" number").columns
            df_cleand[numeric_columns] = df_cleand[numeric_columns].fillna(df[numeric_columns].median())
        elif self.method == 'constant':
             df_cleand = df_cleand.fillna(self.fill_value)
        else:
            logging.warning(f"Không  có giá trị bị thiếu.")
        
        logging.info("Gía trị bị thiếu đã được xử lý.")
        return df_cleand

class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._stratery = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Chiến lược chọn phương pháp xử lý dữ liệu thiếu")
        self._stratery = strategy
    
    def handle_missing_value(self, df: pd.DataFrame):

        logging.info("Thực thi chiến lược xử lý dữ liệu")
        return self._stratery.handle(df)

if __name__ == "__main__":
    pass