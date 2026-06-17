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
