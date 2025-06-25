# 🩺 Wearables Health Monitoring System

A real-time health monitoring web app built with **Streamlit**, powered by **LSTM models** and **live-simulated wearable data**. The system predicts heart risk based on physiological inputs using a trained deep learning model.

## 🚀 Features

- Live streaming of synthetic health data
- LSTM-based heart risk prediction
- Real-time visualization with Streamlit
- Dockerized for easy deployment

## 🗂️ Project Structure

```
wearables_project/
├── analysis/
├── app.py               # Main Streamlit app
├── data/
│   └── heart_data.csv   # Live streamed data
├── models/
│   ├── heart_risk_lstm_model.keras
│   └── scaler_fixed.pkl
├── services/
│   ├── predictor.py         # Loads model + predicts
│   ├── retrain_scaler.py    # (Optional) retrains scaler
│   └── stream_wearable_data.py # Generates live health data
├── requirements.txt
├── Dockerfile
```

## 🛠️ How to Run Locally

```bash
git clone https://github.com/yourusername/wearables-monitoring.git
cd wearables-monitoring
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Run with Docker

```bash
docker build -t wearables-app .
docker run -p 8501:8501 wearables-app
```

## 🔁 Auto Streaming Setup

To simulate live data without typing:
```bash
python services/stream_wearable_data.py
```

You can also automate this to run continuously using a background task, cron job, or `supervisord`.

## 📦 Deployment

To deploy on Streamlit Cloud:

1. Push to a public GitHub repo.
2. Go to https://streamlit.io/cloud and connect your repo.
3. Set the entry point as `app.py`.
4. Note: Streamlit Cloud doesn’t support Docker, only the Python app.

## 👨‍🔬 Model

- Model: LSTM Neural Network
- Input: 24-sequence timesteps of vitals
- Output: Binary risk (0/1) and probability

## ✨ Future Improvements

- Real device integration (e.g., Fitbit, Apple Watch)
- Alert system on high-risk prediction
- User dashboard with history tracking

## 🧠 Author

**Anosh Andrews**  
ML Engineer & Tinkerer  
[GitHub](https://github.com/anoshandrews)

---

_This is a prototype and not intended for real medical use._
