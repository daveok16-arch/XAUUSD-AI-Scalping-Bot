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
