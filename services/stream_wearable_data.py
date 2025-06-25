# stream_wearable_data.py

import time
import random
import pandas as pd
from datetime import datetime
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "heart_data.csv")

def generate_sample(patient_id):
    return {
        "timestamp": datetime.now().isoformat(),
        "person_id": patient_id,
        "heart_rate": random.randint(60, 100),
        "blood_pressure": random.randint(110, 150),  # Systolic only for simplicity
        "oxygen_level": round(random.uniform(95, 100), 2),
        "respiratory_rate": random.randint(12, 20),
        "temperature": round(random.uniform(36.5, 38.5), 1),
        "ecg_signal": round(random.uniform(0.8, 1.2), 2),
        "heart_risk": 0  # Let the model predict this later
    }

def simulate_data_stream(filename=DATA_PATH, interval=5, patient_id="P001"):
    print(f"🔄 Starting continuous health data stream to {filename} every {interval}s...")

    while True:
        sample = generate_sample(patient_id)
        df = pd.DataFrame([sample])

        # Append or create the file
        write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
        df.to_csv(filename, mode='a', header=write_header, index=False)

        print(f"📡 {sample}")
        time.sleep(interval)

# Optional standalone execution
if __name__ == "__main__":
    simulate_data_stream()