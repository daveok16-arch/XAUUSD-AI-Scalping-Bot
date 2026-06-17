
import os
import time
import threading
import requests
import joblib
import yfinance as yf

from src.features.feature_engineering import FeatureEngineer

MODEL_PATH = os.getenv("MODEL_PATH", "./models/xauusd_best_model.joblib")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

model = None
fe = FeatureEngineer()


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Telegram worker model loaded")


def send_signal(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram variables missing")
        return

    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": msg
        },
        timeout=10
    )


def run_prediction():

    if model is None:
        return

    try:
        df = yf.download(
            "GC=F",
            period="10d",
            interval="1h",
            progress=False
        )

        df.columns = [
            c.lower()
            for c in df.columns
        ]

        df = fe.create_all_features(df).dropna()

        X = df.tail(1)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        signal = "BUY 🟢" if pred == 1 else "SELL 🔴"

        send_signal(
            f"""
XAUUSD AI SIGNAL

{signal}

Confidence:
{prob:.2%}

Price:
{float(X['close'].iloc[0])}
"""
        )

    except Exception as e:
        print("Prediction error:", e)


def signal_loop():

    while True:
        run_prediction()
        time.sleep(3600)


def init_worker():

    load_model()

    t = threading.Thread(
        target=signal_loop,
        daemon=True
    )

    t.start()

    print("Telegram signal worker started")
