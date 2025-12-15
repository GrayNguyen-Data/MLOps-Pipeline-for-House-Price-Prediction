from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.visualization_heatmap(df)
        self.visualization_matrix(df)

    @abstractmethod
    def visualization_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualization_matrix(self, df: pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def visualization_heatmap(self, df: pd.DataFrame):
        """Hiển thị ma trận tương quan"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot = True, fmt =".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def visualization_matrix(self, df: pd.DataFrame):
        """Hiển thị ma trận biểu đồ phân tán"""
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features",y=1.02)
        plt.show()

if __name__ == "__main__":
    pass