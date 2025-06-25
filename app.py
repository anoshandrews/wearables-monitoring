import streamlit as st
import pandas as pd
import os 
import time
from services.predictor import predict_risk
from streamlit_autorefresh import st_autorefresh
import threading
import time
from services import stream_wearable_data

# Run only once
if "stream_started" not in st.session_state:
    def run_stream():
        stream_wearable_data.simulate_data_stream()  # define this function inside stream_wearable_data.py
    threading.Thread(target=run_stream, daemon=True).start()
    st.session_state["stream_started"] = True
    

# Refresh every 3 seconds
st_autorefresh(interval=1000, key="datarefresh")


BASE_DIR = os.path.dirname(__file__)  # Directory where app.py lives
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_data.csv")

st.set_page_config(page_title="Heart Risk Monitor", layout="wide")
st.title("🩺 Real-Time Heart Risk Monitor")
st.subheader("Powered by AI and Wearables")

# Optional refresh rate
REFRESH_SEC = 2

df = pd.read_csv(DATA_PATH)

if len(df) < 24:
    st.warning("⏳ Waiting for enough vitals to start prediction (need 24 rows)...")
else:
    seq = df.tail(24).copy()
    seq["heart_risk"] = 0
    label, risk = predict_risk(seq)
    latest = seq.iloc[-1:]
    risk_percent = round(float(risk) * 100, 2)

    st.markdown(f"### 👤 Patient ID: `{latest['person_id'].values[0]}`")
    col1, col2, col3 = st.columns(3)

    col1.metric("❤️ Heart Rate", f"{latest['heart_rate'].values[0]} bpm")
    col1.metric("🩸 Blood Pressure", f"{latest['blood_pressure'].values[0]} mmHg")
    col2.metric("🫁 Oxygen Level", f"{latest['oxygen_level'].values[0]} %")
    col2.metric("🌡️ Temperature", f"{latest['temperature'].values[0]} °C")
    col3.metric("📈 ECG Signal", f"{latest['ecg_signal'].values[0]:.2f}")
    col3.metric("🚨 Heart Risk", f"{risk_percent} %", delta="High" if label else "Low", delta_color="inverse")

    st.progress(min(float(risk), 1.0))

    with st.expander("📊 Vital History (last 10 records)"):
        st.line_chart(df.tail(10).set_index("timestamp")[[
            "heart_rate", "blood_pressure", "oxygen_level", "temperature"
        ]])

# 🔁 Automatically refresh every X seconds
time.sleep(REFRESH_SEC)
st.rerun()