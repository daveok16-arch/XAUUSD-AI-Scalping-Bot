#!/bin/bash

set -e

echo "Creating Telegram module..."

cat > src/telegram_bot.py <<'PY'
import os
import requests
import logging

logger = logging.getLogger(__name__)

def send_telegram_message(message):
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not token or not chat_id:
        logger.warning("Telegram credentials missing")
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message
            },
            timeout=20
        )

        r.raise_for_status()
        logger.info("Telegram message sent")
        return True

    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False
PY


echo "Creating signal worker..."

cat > src/signal_worker.py <<'PY'
import time
import logging

from src.telegram_bot import send_telegram_message
from src.data.data_fetcher import XAUUSDDataFetcher
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

MODEL = None
SCALER = None
FEATURES = []

LAST_SIGNAL = None


def init_worker(model, scaler, features):
    global MODEL, SCALER, FEATURES

    MODEL = model
    SCALER = scaler
    FEATURES = features

    logger.info("Signal worker ready")


def run_prediction():

    global LAST_SIGNAL

    fetcher = XAUUSDDataFetcher()

    df = fetcher.fetch_data(
        period="730d",
        interval="1h"
    )

    engineer = FeatureEngineer()

    df = engineer.create_all_features(df)

    latest = df.iloc[-1:]

    X = latest[FEATURES]

    X_scaled = SCALER.transform(X)

    prediction = MODEL.predict(X_scaled)[0]

    confidence = MODEL.predict_proba(X_scaled)[0][1]


    signal = "BUY" if prediction == 1 else "SELL"


    if signal != LAST_SIGNAL:

        send_telegram_message(
            f"""
🟡 XAUUSD AI SIGNAL

Signal: {signal}

Confidence:
{confidence:.2%}

Timeframe:
1H

AI Model:
XAUUSD Scalping Bot
"""
        )

        LAST_SIGNAL = signal


def signal_loop():

    while True:

        try:
            run_prediction()

        except Exception as e:
            logger.error(f"Signal error: {e}")

        time.sleep(300)
PY


echo "Patching Render gunicorn..."

python - <<'PY'
from pathlib import Path

p = Path("render.yaml")

text = p.read_text()

text = text.replace(
"startCommand: python -m src.api.app",
"startCommand: gunicorn -b 0.0.0.0:$PORT src.api.app:app"
)

p.write_text(text)

print("Render patched")
PY


echo "Injecting worker startup..."

python - <<'PY'
from pathlib import Path

p = Path("src/api/app.py")

text = p.read_text()

inject = '''
from src.signal_worker import init_worker, signal_loop
import threading

init_worker(
    MODEL,
    SCALER,
    FEATURE_NAMES
)

threading.Thread(
    target=signal_loop,
    daemon=True
).start()
'''

marker = "logger.info(f\"Model loaded successfully from {model_path}\")"

if "signal_worker import" not in text:

    text = text.replace(
        marker,
        marker + "\n" + inject
    )

    p.write_text(text)

print("API patched")
PY


echo "Saving changes..."

git add .

git commit -m "Add automated Telegram XAUUSD signal delivery"

git push origin main

echo "DONE - Render will redeploy"
