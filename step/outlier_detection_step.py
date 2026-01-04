from typing import Annotated
import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def outlier_detection_step(
    df: Annotated[pd.DataFrame, "clean_data"],
) -> Annotated[pd.DataFrame, "outlier_removed_data"]:
    """Phát hiện và loại bỏ các giá trị ngoại lai (outliers)."""
    logging.info(f"Bắt đầu bước phát hiện outlier, kích thước DataFrame: {df.shape}")
    
    # 1. Xác định các cột liên tục (Continuous Features)
    # Loại bỏ các cột dạng OHE/Binary (chỉ chứa 0 và 1) bằng cách chọn các cột có số lượng giá trị unique > 2
    continuous_cols = []
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() > 2:
            continuous_cols.append(col)
    
    logging.info(f"Đã tự động chọn {len(continuous_cols)} cột để phát hiện outlier.")

    if not continuous_cols:
        logging.warning("Không tìm thấy cột liên tục nào để xử lý outlier. Bỏ qua bước này.")
        return df

    # 2. Áp dụng phát hiện outlier chỉ trên các cột liên tục
    df_continuous = df[continuous_cols]
    
    # Sử dụng phương pháp Z-Score (ngưỡng = 3)
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    
    # Tạo mask đánh dấu outlier (True nếu là outlier)
    # Ví dụ: df_continuous.shape = (2930, số_cột)
    outliers_mask = outlier_detector.detected_outlier(df_continuous) 
    
    # Một hàng được coi là outlier nếu nó là outlier ở BẤT KỲ cột liên tục nào
    outliers_to_remove = outliers_mask.any(axis=1) 
    
    # Loại bỏ các outlier khỏi DataFrame gốc
    df_cleaned = df[~outliers_to_remove]
    
    logging.info(f"Đã loại bỏ outlier. Số dòng ban đầu: {df.shape[0]}, Số dòng còn lại: {df_cleaned.shape[0]}")
    
    return df_cleaned
