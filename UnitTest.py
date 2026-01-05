from zenml.client import Client
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import sys, io
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
# =========================================================
# 1. SETUP PATH
# =========================================================
ROOT = Path(__file__).resolve().parent
SRC_PATH = str(ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# =========================================================
# 2. CONNECT ZENML
# =========================================================
client = Client()

# =========================================================
# 3. GET LATEST PIPELINE RUN
# =========================================================
runs = client.list_pipeline_runs()
runs = [r for r in runs if r.pipeline.name == "ml_pipeline"]

if not runs:
    raise ValueError("Không tìm thấy pipeline run cho 'ml_pipeline'")

pipeline_run = max(runs, key=lambda r: r.created)
print(f"Pipeline run: {pipeline_run.name}")

# =========================================================
# 4. LOAD MODEL
# =========================================================
model_artifact = (
    pipeline_run
    .steps["model_building_step"]
    .outputs["sklearn_pipeline"][0]
)
model = model_artifact.load()
print("Load Stacking model thành công")

# =========================================================
# 5. LOAD DATASET (X_test, y_test)
# =========================================================
X_test = (
    pipeline_run
    .steps["data_splitter_step"]
    .outputs["X_test"][0]
    .load()
)

y_test = (
    pipeline_run
    .steps["data_splitter_step"]
    .outputs["y_test"][0]
    .load()
)

# =========================================================
# 6. CREATE NEW SAMPLE (MEAN / MODE)
# =========================================================
def create_mean_sample(X: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            row[col] = X[col].mean()
        else:
            row[col] = X[col].mode()[0]
    return pd.DataFrame([row])

X_new = create_mean_sample(X_test)

print("\n================ NEW SAMPLE ==================")
print(X_new)

# =========================================================
# 7. PREDICTION
# =========================================================
y_pred_log = float(model.predict(X_new)[0])
y_pred_real = np.exp(y_pred_log)

# =========================================================
# 8. GROUND TRUTH REFERENCE (MEAN y_test)
# =========================================================
y_true_log_mean = float(y_test.mean())
y_true_real_mean = np.exp(y_true_log_mean)

# =========================================================
# 9. FINAL OUTPUT
# =========================================================
print("\n================= KẾT QUẢ =================")
print(f"Giá nhà dự đoán (log-scale): {y_pred_log:.6f}")
print(f"Giá nhà dự đoán (ước tính):  {y_pred_real:,.0f}")

print("------------------------------------------")
print(f"Giá nhà thực tế TB (log):    {y_true_log_mean:.6f}")
print(f"Giá nhà thực tế TB (thật):   {y_true_real_mean:,.0f}")
print("==========================================")
