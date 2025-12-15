from abc import ABC, abstractmethod
import pandas as pd

"""Lớp Product - định nghĩa lớp inspect cho tất cả class con sài"""
class DataIngestionStrategy(ABC):
    def inspect(self, df: pd.DataFrame):
        pass

"""Class xem cấu trúc của DataFrame"""
class DataTypeInspectionStrategy(DataIngestionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nCấu trúc của DataFrame")
        print(df.info())

"""Class thống kê mô tả"""
class SummarryStatisticsInspectionStrategy(DataIngestionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nThống kê tóm tắt DataFrame cho cột số")
        print(df.describe())

        print("\nThống kê tóm tắt DataFrame cho cột object/category")
        print(df.describe(include="O"))

class DataInspector:
    def __init__(self, strategy: DataIngestionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataIngestionStrategy):
        self._strategy = strategy

    def execute_inspector(self, df: pd.DataFrame):
        self._strategy.inspect(df)

if __name__ == "__main__":
    pass
