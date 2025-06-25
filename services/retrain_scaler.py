import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load original training data
df = pd.read_csv("/Users/anoshandrews/Documents/Machine Learning/live_health_monitoring/data/heart_data.csv")

# These are your actual input features
FEATURES = [
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "respiratory_rate",
    "temperature",
    "ecg_signal"
]

X = df[FEATURES]

# Refit scaler only on input features
scaler = StandardScaler()
scaler.fit(X)

# Save it
joblib.dump(scaler, "/Users/anoshandrews/Documents/Machine Learning/live_health_monitoring/scaler_fixed.pkl")
print("✅ New scaler trained and saved as scaler_fixed.pkl")