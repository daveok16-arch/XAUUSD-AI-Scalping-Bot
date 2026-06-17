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

FEATURES_FILE = "./models/xauusd_best_model_features.txt"
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))
PREDICTION_INTERVAL = int(os.getenv("PREDICTION_INTERVAL", "300"))

EXPECTED_FEATURES = []

if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE) as f:
        EXPECTED_FEATURES = [x.strip() for x in f if x.strip()]

model = None
fe = FeatureEngineer()


def send_signal(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram variables missing")
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": msg,
            },
            timeout=15,
        )
    except Exception as e:
        print("Telegram send error:", e)


def load_model():
    global model

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Telegram worker model loaded")
        send_signal("XAUUSD AI Bot started successfully ✅")
    else:
        print("Model not found:", MODEL_PATH)


def run_prediction():

    if model is None:
        return

    try:
        df = yf.download(
            "GC=F",
            period="10d",
            interval="1h",
            progress=False,
        )

        df.columns = [str(c).lower() for c in df.columns]

        df = fe.create_all_features(df).dropna()

        if len(df) == 0:
            print("No feature rows generated")
            return

        missing = [
            f for f in EXPECTED_FEATURES
            if f not in df.columns
        ]

        for col in missing:
            df[col] = 0

        X = df[EXPECTED_FEATURES].tail(1)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        confidence = prob if pred == 1 else (1 - prob)

        if confidence < MIN_CONFIDENCE:
            print(
                f"Signal filtered: confidence={confidence:.2%}"
            )
            return

        signal = "BUY 🟢" if pred == 1 else "SELL 🔴"

        price = float(df["close"].iloc[-1])

        send_signal(
            f"""XAUUSD AI SIGNAL

{signal}

Confidence: {confidence:.2%}

Price: {price}

Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        )

        print(
            f"Signal sent: {signal} confidence={confidence:.2%}"
        )

    except Exception as e:
        print("Prediction error:", e)


def signal_loop():

    while True:
        run_prediction()
        time.sleep(PREDICTION_INTERVAL)


def init_worker():

    load_model()

    t = threading.Thread(
        target=signal_loop,
        daemon=True,
    )

    t.start()

    print("Telegram signal worker started")
