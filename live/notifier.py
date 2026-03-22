# live/notifier.py
"""Bildirim sistemi — Telegram entegrasyonu."""
import os
import logging

logger = logging.getLogger(__name__)


class Notifier:
    """Temel bildirim sınıfı (no-op)."""
    def send(self, message: str):
        pass


class TelegramNotifier(Notifier):
    """
    Telegram bot ile bildirim gönderir.

    Env variables:
        TELEGRAM_BOT_TOKEN: Bot token (@BotFather'dan alınır)
        TELEGRAM_CHAT_ID: Mesaj gönderilecek chat ID
    """

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.warning(
                "[Telegram] TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID eksik. "
                "Bildirimler devre dışı."
            )

    def send(self, message: str):
        if not self.enabled:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }, timeout=10)
            if not resp.ok:
                logger.warning(f"[Telegram] Gönderim hatası: {resp.status_code} {resp.text}")
        except ImportError:
            logger.warning("[Telegram] 'requests' paketi yüklü değil: pip install requests")
        except Exception as e:
            logger.warning(f"[Telegram] Bildirim hatası: {e}")


def get_notifier() -> Notifier:
    """Ortam değişkenlerine göre uygun notifier döndür."""
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        return TelegramNotifier()
    return Notifier()
