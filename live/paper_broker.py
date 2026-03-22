# live/paper_broker.py
"""Sanal broker — gerçek para kullanmadan trade simülasyonu."""
import logging
import uuid
from datetime import datetime

from live.broker_base import BrokerBase

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """
    Paper trading broker.
    - Sanal nakit ve pozisyon takibi
    - Pending stop/limit emirleri bar bazlı kontrol
    - Komisyon hesabı
    """

    def __init__(self, cash: float = 10000.0, commission: float = 0.001):
        self.cash = cash
        self.commission = commission
        self.position: dict | None = None  # {symbol, side, size, entry_price}
        self.pending_orders: list[dict] = []
        self.trade_log: list[dict] = []

    def get_balance(self) -> float:
        return self.cash

    def get_equity(self, current_price: float) -> float:
        equity = self.cash
        if self.position:
            pnl = self._unrealized_pnl(current_price)
            equity += pnl
        return equity

    def get_position(self, symbol: str) -> dict | None:
        if self.position and self.position.get("symbol") == symbol:
            return self.position
        return None

    def place_market_order(self, symbol: str, side: str, size: float) -> dict:
        # Simüle market fill — mevcut fiyatla (engine tarafından sağlanır)
        order_id = str(uuid.uuid4())[:8]
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "type": "market",
            "status": "pending",
        }
        # Market emirler anında fill olur — engine _execute_fill çağırmalı
        logger.info(f"[Paper] Market {side} {size:.6f} {symbol} (id={order_id})")
        return order

    def place_stop_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        order_id = str(uuid.uuid4())[:8]
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "type": "stop",
            "status": "open",
        }
        self.pending_orders.append(order)
        logger.info(f"[Paper] Stop {side} @ {price:.2f} (id={order_id})")
        return order

    def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        order_id = str(uuid.uuid4())[:8]
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "type": "limit",
            "status": "open",
        }
        self.pending_orders.append(order)
        logger.info(f"[Paper] Limit {side} @ {price:.2f} (id={order_id})")
        return order

    def cancel_order(self, order_id: str) -> bool:
        for i, o in enumerate(self.pending_orders):
            if o["order_id"] == order_id:
                self.pending_orders.pop(i)
                logger.info(f"[Paper] Emir iptal: {order_id}")
                return True
        return False

    def cancel_all_orders(self, symbol: str) -> int:
        before = len(self.pending_orders)
        self.pending_orders = [
            o for o in self.pending_orders if o.get("symbol") != symbol
        ]
        cancelled = before - len(self.pending_orders)
        if cancelled:
            logger.info(f"[Paper] {cancelled} emir iptal edildi ({symbol})")
        return cancelled

    def get_open_orders(self, symbol: str) -> list:
        return [o for o in self.pending_orders if o.get("symbol") == symbol]

    def execute_fill(self, symbol: str, side: str, size: float, fill_price: float):
        """Market emri veya tetiklenen stop/limit emrini uygula."""
        cost = size * fill_price
        comm = cost * self.commission

        if side == "buy":
            if self.position and self.position["side"] == "short":
                # Short kapatma
                pnl = (self.position["entry_price"] - fill_price) * self.position["size"]
                self.cash += pnl - comm
                self._log_trade("short", self.position["entry_price"], fill_price, pnl - comm)
                self.position = None
            else:
                # Long açma
                self.cash -= comm
                self.position = {
                    "symbol": symbol,
                    "side": "long",
                    "size": size,
                    "entry_price": fill_price,
                }
        elif side == "sell":
            if self.position and self.position["side"] == "long":
                # Long kapatma
                pnl = (fill_price - self.position["entry_price"]) * self.position["size"]
                self.cash += pnl - comm
                self._log_trade("long", self.position["entry_price"], fill_price, pnl - comm)
                self.position = None
            else:
                # Short açma
                self.cash -= comm
                self.position = {
                    "symbol": symbol,
                    "side": "short",
                    "size": size,
                    "entry_price": fill_price,
                }

    def check_pending_orders(self, symbol: str, bar: dict):
        """
        Bar verisiyle pending emirleri kontrol et.
        bar: {'high': float, 'low': float, 'close': float} veya pd.Series
        """
        high = float(bar.get("high", bar.get("High", 0)))
        low = float(bar.get("low", bar.get("Low", 0)))

        triggered = []
        for order in list(self.pending_orders):
            if order["symbol"] != symbol or order["status"] != "open":
                continue

            fill_price = None

            if order["type"] == "stop":
                # Stop sell: fiyat stop'un altına düşerse
                if order["side"] == "sell" and low <= order["price"]:
                    fill_price = order["price"]
                # Stop buy: fiyat stop'un üstüne çıkarsa
                elif order["side"] == "buy" and high >= order["price"]:
                    fill_price = order["price"]

            elif order["type"] == "limit":
                # Limit sell: fiyat limitin üstüne çıkarsa
                if order["side"] == "sell" and high >= order["price"]:
                    fill_price = order["price"]
                # Limit buy: fiyat limitin altına düşerse
                elif order["side"] == "buy" and low <= order["price"]:
                    fill_price = order["price"]

            if fill_price is not None:
                triggered.append(order)
                self.execute_fill(symbol, order["side"], order["size"], fill_price)
                logger.info(
                    f"[Paper] {order['type'].upper()} {order['side']} tetiklendi "
                    f"@ {fill_price:.2f} (id={order['order_id']})"
                )

        # Tetiklenen emirleri listeden çıkar
        triggered_ids = {o["order_id"] for o in triggered}
        self.pending_orders = [
            o for o in self.pending_orders if o["order_id"] not in triggered_ids
        ]

        # Pozisyon kapandıysa kalan emirleri iptal et (SL tetiklendi → TP iptal, vs.)
        if self.position is None and triggered:
            self.cancel_all_orders(symbol)

    def _unrealized_pnl(self, current_price: float) -> float:
        if not self.position:
            return 0.0
        if self.position["side"] == "long":
            return (current_price - self.position["entry_price"]) * self.position["size"]
        else:
            return (self.position["entry_price"] - current_price) * self.position["size"]

    def _log_trade(self, side: str, entry: float, exit_px: float, pnl: float):
        trade = {
            "dt": datetime.now().isoformat(),
            "side": side,
            "entry_price": entry,
            "exit_price": exit_px,
            "pnl": pnl,
        }
        self.trade_log.append(trade)
        logger.info(
            f"[Paper] TRADE {side.upper()} closed: "
            f"entry={entry:.2f} exit={exit_px:.2f} pnl={pnl:.2f}"
        )
