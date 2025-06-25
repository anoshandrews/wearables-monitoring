import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os 

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "heart_risk_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler_fixed.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

SEQUENCE_LENGTH = 24

MODEL_FEATURES = [
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "respiratory_rate",
    "temperature",
    "ecg_signal",
    "heart_risk"
]

SCALER_FEATURES = [
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "respiratory_rate",
    "temperature",
    "ecg_signal"
]

def predict_risk(df: pd.DataFrame):
    if not all(col in df.columns for col in MODEL_FEATURES):
        raise ValueError(f"Missing required features. Found: {list(df.columns)}")

    if len(df) < SEQUENCE_LENGTH:
        last_row = df.iloc[-1:]
        padding = pd.concat([last_row] * (SEQUENCE_LENGTH - len(df)), ignore_index=True)
        df = pd.concat([padding, df], ignore_index=True)

    seq = df[MODEL_FEATURES].tail(SEQUENCE_LENGTH).copy()

    # Scale only the scaler features
    X_to_scale = seq[SCALER_FEATURES]
    X_scaled = scaler.transform(X_to_scale)

    # Replace scaled features back into the dataframe
    for i, col in enumerate(SCALER_FEATURES):
        seq[col] = X_scaled[:, i]

    # Final input shape
    X_final = seq.values.reshape(1, SEQUENCE_LENGTH, len(MODEL_FEATURES))

    risk_score = model.predict(X_final)[0][0]
    label = int(risk_score > 0.5)
    return label, risk_score