# live/engine.py
"""Ana trading döngüsü — veri çek, sinyal üret, emir ver."""
import os
import sys
import time
import signal
import logging
from datetime import datetime

import config
from live.signal_generator import SignalGenerator
from live.broker_base import BrokerBase
from live.paper_broker import PaperBroker
from live.state import State
from live.notifier import Notifier, get_notifier
from utils.data_provider import DataProvider

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Canlı/Paper trading döngüsü.

    Döngü:
    1. Yeni mum verisi çek (trade + trend)
    2. Pending emirleri kontrol et (SL/TP)
    3. İndikatörleri hesapla
    4. Pozisyon varsa çıkış sinyali kontrol et
    5. Pozisyon yoksa giriş sinyali kontrol et
    6. State kaydet
    """

    def __init__(
        self,
        broker: BrokerBase,
        signal_gen: SignalGenerator,
        provider: DataProvider,
        symbol: str,
        timeframe_trade: str,
        timeframe_trend: str,
        notifier: Notifier | None = None,
        state_path: str = "data/state.json",
        poll_interval: int = 60,
        warmup_bars: int = 300,
        risk_per_trade: float = 0.005,
    ):
        self.broker = broker
        self.signal_gen = signal_gen
        self.provider = provider
        self.symbol = symbol
        self.timeframe_trade = timeframe_trade
        self.timeframe_trend = timeframe_trend
        self.notifier = notifier or get_notifier()
        self.state_path = state_path
        self.poll_interval = poll_interval
        self.warmup_bars = warmup_bars
        self.risk_per_trade = risk_per_trade

        self._running = False
        self.state = State()
        self.tick_count = 0

    def run(self):
        """Ana döngüyü başlat."""
        self.state = State.load(self.state_path)

        # Paper broker state yükle
        if isinstance(self.broker, PaperBroker):
            self.broker.cash = self.state.cash
            if self.state.position:
                self.broker.position = self.state.position

        self._running = True
        self._setup_signal_handlers()

        mode_str = "PAPER" if isinstance(self.broker, PaperBroker) else "LIVE"
        start_msg = (
            f"Bot başlatıldı [{mode_str}]\n"
            f"Sembol: {self.symbol}\n"
            f"Trade TF: {self.timeframe_trade}, Trend TF: {self.timeframe_trend}\n"
            f"Poll: {self.poll_interval}s"
        )
        logger.info(start_msg)
        self.notifier.send(start_msg)
        print(f"\n{'='*50}")
        print(f"  {mode_str} TRADING - {self.symbol}")
        print(f"  Çıkmak için Ctrl+C")
        print(f"{'='*50}\n")

        last_bar_time = self.state.last_bar_time

        while self._running:
            try:
                self.tick_count += 1

                # 1. Veri çek
                df_trade = self.provider.fetch_ohlcv(
                    self.symbol, self.timeframe_trade,
                    self.warmup_bars, self.warmup_bars,
                )
                trend_bars = max(100, self.warmup_bars // 12)
                df_trend = self.provider.fetch_ohlcv(
                    self.symbol, self.timeframe_trend,
                    trend_bars, trend_bars,
                )

                if df_trade is None or df_trade.empty:
                    logger.warning("Veri çekilemedi, bekleniyor...")
                    time.sleep(30)
                    continue

                latest_bar_time = str(df_trade.index[-1])
                current_price = float(df_trade["close"].iloc[-1])

                # 2. Yeni mum mu?
                if last_bar_time is not None and latest_bar_time == last_bar_time:
                    time.sleep(self.poll_interval)
                    continue

                last_bar_time = latest_bar_time
                current_dt = df_trade.index[-1]

                # Durum yazdır
                equity = self.broker.get_equity(current_price)
                pos = self.broker.get_position(self.symbol)
                pos_str = f"{pos['side']} {pos['size']:.6f}" if pos else "YOK"
                print(
                    f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                    f"{self.symbol} = {current_price:.2f} | "
                    f"Equity: {equity:.2f} | Poz: {pos_str}"
                )

                # 3. Pending emirleri kontrol et (paper)
                if isinstance(self.broker, PaperBroker):
                    latest_bar = df_trade.iloc[-1].to_dict()
                    self.broker.check_pending_orders(self.symbol, latest_bar)

                # 4. İndikatörleri hesapla
                indicators = self.signal_gen.compute_indicators(df_trade, df_trend)

                # 5. Rejim (basit: trend verisinden)
                current_regime = None  # v1: rejim filtresi kapalı (pre-trained model sonra)

                # 6. Pozisyon yönetimi
                position = self.broker.get_position(self.symbol)

                if position:
                    # Bars held hesapla
                    bars_held = self.state.bars_since_last_trade

                    # BE stop kontrolü
                    self._check_breakeven(position, current_price)

                    # Çıkış sinyali
                    exit_reason = self.signal_gen.should_exit(
                        indicators, position["side"], bars_held, current_regime,
                    )
                    if exit_reason:
                        self._close_position(position, current_price, exit_reason)

                else:
                    # Giriş sinyali
                    sig = self.signal_gen.get_signal(
                        indicators, current_regime,
                        self.state.bars_since_last_trade, current_dt,
                    )
                    if sig.direction:
                        self._open_position(sig, current_price, equity)

                # 7. State güncelle
                self.state.last_bar_time = last_bar_time
                self.state.bars_since_last_trade += 1
                if isinstance(self.broker, PaperBroker):
                    self.state.cash = self.broker.cash
                    self.state.position = self.broker.position
                self.state.save(self.state_path)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Engine hatası: {e}", exc_info=True)
                self.notifier.send(f"HATA: {e}")
                time.sleep(60)

            time.sleep(self.poll_interval)

        self._shutdown()

    def _open_position(self, sig, current_price: float, equity: float):
        """Sinyal doğrultusunda pozisyon aç."""
        # ATR bazlı pozisyon boyutu
        sl_distance = abs(current_price - sig.sl_price)
        if sl_distance < 1e-9:
            sl_distance = current_price * 0.02  # fallback

        risk_amount = equity * self.risk_per_trade
        size = risk_amount / sl_distance

        side = "buy" if sig.direction == "long" else "sell"

        # Market emir
        if isinstance(self.broker, PaperBroker):
            self.broker.execute_fill(self.symbol, side, size, current_price)
        else:
            result = self.broker.place_market_order(self.symbol, side, size)
            current_price = result.get("filled_price", current_price)

        # SL/TP emirleri
        sl_side = "sell" if sig.direction == "long" else "buy"
        tp_side = sl_side

        sl_order = self.broker.place_stop_order(self.symbol, sl_side, size, sig.sl_price)
        tp_order = self.broker.place_limit_order(self.symbol, tp_side, size, sig.tp_price)

        # State güncelle
        self.state.position = {
            "side": sig.direction,
            "size": size,
            "entry_price": current_price,
        }
        self.state.entry_price = current_price
        self.state.entry_bar = self.tick_count
        self.state.bars_since_last_trade = 0
        self.state.pending_sl_id = sl_order.get("order_id")
        self.state.pending_tp_id = tp_order.get("order_id")
        self.state.be_stop_active = False

        msg = (
            f"ENTRY {sig.direction.upper()} {self.symbol}\n"
            f"Fiyat: {current_price:.2f}\n"
            f"Size: {size:.6f}\n"
            f"SL: {sig.sl_price:.2f} | TP: {sig.tp_price:.2f}\n"
            f"Neden: {sig.reason}"
        )
        logger.info(msg)
        self.notifier.send(msg)
        print(f"  >>> {msg}")

    def _close_position(self, position: dict, current_price: float, reason: str):
        """Pozisyonu kapat."""
        size = position["size"]
        side = "sell" if position["side"] == "long" else "buy"

        # Pending emirleri iptal
        self.broker.cancel_all_orders(self.symbol)

        # Market çıkış
        if isinstance(self.broker, PaperBroker):
            self.broker.execute_fill(self.symbol, side, size, current_price)
        else:
            self.broker.place_market_order(self.symbol, side, size)

        entry_px = self.state.entry_price or position.get("entry_price", 0)
        if position["side"] == "long":
            pnl = (current_price - entry_px) * size
        else:
            pnl = (entry_px - current_price) * size

        msg = (
            f"EXIT {position['side'].upper()} {self.symbol}\n"
            f"Giriş: {entry_px:.2f} → Çıkış: {current_price:.2f}\n"
            f"PnL: {pnl:.2f}\n"
            f"Neden: {reason}"
        )
        logger.info(msg)
        self.notifier.send(msg)
        print(f"  <<< {msg}")

        self.state.reset_position()
        self.state.trade_count += 1

    def _check_breakeven(self, position: dict, current_price: float):
        """1.5R kârda SL'yi breakeven'a taşı."""
        if self.state.be_stop_active:
            return

        entry = self.state.entry_price
        if not entry:
            return

        risk = abs(entry - (position.get("sl_price", entry * 0.98)))
        if risk < 1e-9:
            return

        if position["side"] == "long" and current_price >= entry + 1.5 * risk:
            # SL'yi entry'ye taşı
            self.broker.cancel_all_orders(self.symbol)
            size = position["size"]
            self.broker.place_stop_order(self.symbol, "sell", size, entry)
            self.state.be_stop_active = True
            logger.info(f"[BE] SL breakeven'a taşındı: {entry:.2f}")

        elif position["side"] == "short" and current_price <= entry - 1.5 * risk:
            self.broker.cancel_all_orders(self.symbol)
            size = position["size"]
            self.broker.place_stop_order(self.symbol, "buy", size, entry)
            self.state.be_stop_active = True
            logger.info(f"[BE] SL breakeven'a taşındı: {entry:.2f}")

    def _setup_signal_handlers(self):
        """Graceful shutdown için sinyal yakala."""
        def _handler(signum, frame):
            print("\nKapatılıyor...")
            self._running = False

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def _shutdown(self):
        """Temiz kapanış."""
        self.state.save(self.state_path)

        equity = self.broker.get_equity(0)
        msg = (
            f"Bot kapatıldı [{self.symbol}]\n"
            f"Toplam trade: {self.state.trade_count}\n"
            f"Bakiye: {equity:.2f}"
        )
        logger.info(msg)
        self.notifier.send(msg)
        print(f"\n{msg}")

        # Paper broker trade log'unu yazdır
        if isinstance(self.broker, PaperBroker) and self.broker.trade_log:
            print(f"\n--- Trade Geçmişi ({len(self.broker.trade_log)} trade) ---")
            for t in self.broker.trade_log:
                print(
                    f"  {t['dt']} | {t['side'].upper()} | "
                    f"Giriş: {t['entry_price']:.2f} → Çıkış: {t['exit_price']:.2f} | "
                    f"PnL: {t['pnl']:.2f}"
                )
