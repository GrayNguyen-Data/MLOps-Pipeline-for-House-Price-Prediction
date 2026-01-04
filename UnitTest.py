from zenml.client import Client
import pandas as pd
import sys
from pathlib import Path

# Thêm src vào sys.path để load model không lỗi ModuleNotFoundError
ROOT = Path(__file__).resolve().parent
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

client = Client()

# Lấy tất cả pipeline run của pipeline "ml_pipeline"
runs = client.list_pipeline_runs()

# Lấy run gần nhất của pipeline này
runs_of_pipeline = [r for r in runs if r.pipeline.name == "ml_pipeline"]
if not runs_of_pipeline:
    raise ValueError("Không tìm thấy pipeline run cho 'ml_pipeline'.")

pipeline_run = max(runs_of_pipeline, key=lambda r: r.created)

# Lấy dữ liệu X_test và y_test
X_test_artifact = pipeline_run.steps["data_splitter_step"].outputs["X_test"][0]
y_test_artifact = pipeline_run.steps["data_splitter_step"].outputs["y_test"][0]

X_test = X_test_artifact.load()
y_test = y_test_artifact.load()

# Lấy trained model từ step model_building_step
model_artifact = pipeline_run.steps["model_building_step"].outputs["sklearn_pipeline"][0]
model = model_artifact.load()

# Dự đoán trên X_test
y_pred_existing = model.predict(X_test)

# Tạo row test mới
columns = X_test.columns
new_row = {}
for col in columns:
    if pd.api.types.is_numeric_dtype(X_test[col]):
        new_row[col] = X_test[col].mean()
    else:
        new_row[col] = X_test[col].mode()[0]

X_new_test = pd.DataFrame([new_row])
y_pred_new = model.predict(X_new_test)

# In kết quả
print("=== Kết quả dự đoán trên X_test gốc ===")
for i in range(len(X_test)):
    print(f"X_test[{i}]: Dự đoán = {y_pred_existing[i]}, Thật = {y_test.iloc[i]}")

print("\n=== Kết quả dự đoán row test mới ===")
print("X_new_test:")
print(X_new_test)
print("Dự đoán:", y_pred_new[0])
