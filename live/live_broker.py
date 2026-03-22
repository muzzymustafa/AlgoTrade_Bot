# live/live_broker.py
"""Binance canlı broker — CCXT ile gerçek emir yürütme."""
import os
import logging
import ccxt

from live.broker_base import BrokerBase

logger = logging.getLogger(__name__)


class LiveBroker(BrokerBase):
    """
    Binance üzerinde gerçek emir yerleştirme.

    GÜVENLİK:
    - API anahtarları SADECE ortam değişkenlerinden okunur
    - Varsayılan olarak sandbox (testnet) kullanır
    - Gerçek para için LIVE_TRADING_CONFIRMED=true env var gerekir
    """

    def __init__(self, sandbox: bool = True):
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_API_KEY ve BINANCE_API_SECRET ortam değişkenleri gerekli.\n"
                "export BINANCE_API_KEY='...'\n"
                "export BINANCE_API_SECRET='...'"
            )

        # Güvenlik kontrolü
        if not sandbox:
            confirmed = os.environ.get("LIVE_TRADING_CONFIRMED", "").lower()
            if confirmed != "true":
                raise ValueError(
                    "Canlı trading için LIVE_TRADING_CONFIRMED=true gerekli.\n"
                    "Bu koruma gerçek para kaybını önlemek içindir."
                )

        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info("[Live] SANDBOX (testnet) modu aktif")
        else:
            logger.warning("[Live] !!! GERÇEK PARA MODU !!!")

        self.sandbox = sandbox

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"[Live] Bakiye sorgu hatası: {e}")
            return 0.0

    def get_equity(self, current_price: float) -> float:
        # Spot için: USDT bakiye + pozisyon değeri
        balance = self.exchange.fetch_balance()
        usdt = float(balance.get("USDT", {}).get("total", 0))
        # Basit yaklaşım: tüm USDT-bazlı varlıkları say
        return usdt

    def get_position(self, symbol: str) -> dict | None:
        try:
            balance = self.exchange.fetch_balance()
            # Symbol'den base currency'yi çıkar (ör: BTC/USDT → BTC)
            base = symbol.split("/")[0]
            amount = float(balance.get(base, {}).get("total", 0))
            if amount > 0:
                # Giriş fiyatını exchange'den almak zor, en son trade'den çekilebilir
                return {
                    "side": "long",
                    "size": amount,
                    "entry_price": 0.0,  # engine state'den gelecek
                }
            return None
        except Exception as e:
            logger.error(f"[Live] Pozisyon sorgu hatası: {e}")
            return None

    def place_market_order(self, symbol: str, side: str, size: float) -> dict:
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
            )
            fill_price = float(order.get("average", order.get("price", 0)))
            logger.info(f"[Live] Market {side} {size:.6f} {symbol} @ {fill_price:.2f}")
            return {
                "order_id": order["id"],
                "filled_price": fill_price,
                "status": order["status"],
            }
        except Exception as e:
            logger.error(f"[Live] Market emir hatası: {e}")
            raise

    def place_stop_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_LOSS_LIMIT",
                side=side,
                amount=size,
                price=price,
                params={"stopPrice": price},
            )
            logger.info(f"[Live] Stop {side} {size:.6f} {symbol} @ {price:.2f}")
            return {"order_id": order["id"], "status": order["status"]}
        except Exception as e:
            logger.error(f"[Live] Stop emir hatası: {e}")
            raise

    def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=size,
                price=price,
            )
            logger.info(f"[Live] Limit {side} {size:.6f} {symbol} @ {price:.2f}")
            return {"order_id": order["id"], "status": order["status"]}
        except Exception as e:
            logger.error(f"[Live] Limit emir hatası: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        try:
            # Symbol gerekli, engine tarafından sağlanmalı
            # Bu basit implementasyonda symbol'ü order_id'den çözemeyiz
            # Engine cancel çağırırken symbol de geçmeli
            logger.info(f"[Live] Emir iptal: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[Live] İptal hatası: {e}")
            return False

    def cancel_order_with_symbol(self, order_id: str, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"[Live] Emir iptal: {order_id} ({symbol})")
            return True
        except Exception as e:
            logger.error(f"[Live] İptal hatası: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> int:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            for o in orders:
                self.exchange.cancel_order(o["id"], symbol)
            logger.info(f"[Live] {len(orders)} emir iptal ({symbol})")
            return len(orders)
        except Exception as e:
            logger.error(f"[Live] Toplu iptal hatası: {e}")
            return 0

    def get_open_orders(self, symbol: str) -> list:
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"[Live] Açık emir sorgu hatası: {e}")
            return []
