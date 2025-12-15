from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """feature là tên cột cần phân tích"""
        pass

class  NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Ta sẽ vẽ biểu đồ histogram cùng với cái đường kẻ KDE để xemn sự phân bố của dữ liệu"""
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=30 )  
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Vẽ biểu đồ BarPlot"""
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature, data = df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()        

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df,feature)

if __name__ =="__main__":
    pass
   
