import streamlit as st
import subprocess, sys, os, glob, re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

st.title("House Price MLOps — Pipeline GUI (with UnitTest)")

def run_and_stream(cmd, cwd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out_area = st.empty()
    out = ""
    if proc.stdout is not None:
        for line in proc.stdout:
            out += line
            out_area.text(out)
    proc.wait()
    return proc.returncode, out

st.markdown("### Actions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Run full pipeline"):
        st.info("Running pipeline — streaming output below.")
        code, output = run_and_stream([sys.executable, "run_pipeline.py"], cwd=ROOT)
        if code == 0:
            st.success("Pipeline finished successfully.")
        else:
            st.error(f"Pipeline exited with code {code}.")

with col2:
    if st.button("Show latest logs"):
        logs = sorted(glob.glob(str(ROOT / "logs" / "*.txt")), key=os.path.getmtime, reverse=True)
        if not logs:
            st.warning("No logs found in logs/")
        else:
            selected = st.selectbox("Choose log file", logs, index=0)
            with open(selected, "r", encoding="utf-8", errors="replace") as f:
                st.code(f.read())

st.markdown("### Unit Test")
st.write("This runs your `UnitTest.py` and shows stdout/stderr. `UnitTest.py` requires ZenML artifacts from a finished pipeline run.")
if st.button("Run UnitTest.py"):
    test_path = ROOT / "UnitTest.py"
    if not test_path.exists():
        st.error("UnitTest.py not found at repository root.")
    else:
        st.info("Running UnitTest.py — output will stream below.")
        code, out = run_and_stream([sys.executable, str(test_path)], cwd=ROOT)
        if code == 0:
            st.success("UnitTest.py finished successfully.")
        else:
            st.error(f"UnitTest.py exited with code {code}.")
        m = re.search(r"Giá nhà dự đoán \(ước tính\):\s*([0-9,]+)", out)
        if m:
            st.metric("Predicted price (approx)", m.group(1))

st.markdown("### Results CSVs")
csvs = sorted(glob.glob(str(ROOT / "results" / "*.csv")))
if csvs:
    chosen = st.selectbox("Choose result CSV", csvs)
    if chosen:
        df = pd.read_csv(chosen)
        st.write(f"Preview of {os.path.basename(chosen)} ({len(df)} rows)")
        st.dataframe(df.head(200))
else:
    st.info("No CSVs found in results/")

st.markdown("### Figures")
figs = sorted(glob.glob(str(ROOT / "figures" / "*.*")))
if figs:
    for f in figs:
        try:
            st.image(f, caption=os.path.basename(f), use_column_width=True)
        except Exception:
            st.write(f"Cannot render {f}")
else:
    st.info("No files in figures/")

st.markdown("### Quick utilities")
if st.button("Print EDA.ipynb path"):
    st.write(str(ROOT / "analysis" / "EDA.ipynb"))