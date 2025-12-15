from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MissingValuesAnalysisTemplate(ABC):
    """Định nghĩa các bước step by step cho quy trình"""
    def analyze(self, df: pd.DataFrame):
        """Tìm các giá trị missing trước rồi đến vẽ biểu đồ trực quan các giá trị missing"""
        self.identity_missing_values(df)
        self.visualize_missing_values(df)
    
    @abstractmethod
    def identity_missing_values(self, df: pd.DataFrame):
        """Định nghĩa cho các lớp con override để tìm các giá trị missing"""
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df:pd.DataFrame):
        """Định nghĩa cho các lớp con"""
        pass

""""Tới lớp concrete class -> cài đặt cụ thể cách tìm missing và vẽ biểu đồ"""
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identity_missing_values(self, df: pd.DataFrame):
        """In ra giá trị missing của các cột"""
        print("\nSố giá trị bị thiếu của cột:")
        missing_value = df.isnull().sum()
        print(missing_value[missing_value>0])
    
    def visualize_missing_values(self, df: pd. DataFrame):
        """Vẽ biểu đồ heatmap với những giá trị bị thiếu"""
        print("\n Vẽ biểu đồ HeatMap giá trị bị thiếu....")
        plt.figure(figsize = (12,8))
        sns.heatmap(df.isnull(), cbar = False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
    
if __name__ == "__main__":
    pass