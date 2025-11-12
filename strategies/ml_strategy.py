# === ml_strategy.py (v38) ===
import backtrader as bt
from statsmodels.tsa.arima.model import ARIMA
import backtrader.indicators as btind
import pandas as pd
import numpy as np
import warnings, os, csv
from datetime import datetime

warnings.filterwarnings("ignore")


# ==============================================================
#            Güvenli SMA (Backfill ve warmup dostu)
# ==============================================================
class SafeSMA(bt.Indicator):
    lines = ("sma",)
    params = (("period", 10),)
    plotinfo = dict(plot=False)

    def __init__(self):
        self.addminperiod(int(self.p.period))

    def next(self):
        n = int(self.p.period)
        s = 0.0
        for i in range(n):
            s += float(self.data.close[-i])
        self.lines.sma[0] = s / n


# ==============================================================
#                   Ana Strateji (v38)
#   - Stop-Entry (swing ± ATR*mult) + zaman aşımı + market fallback
#   - OTR (rejime göre ATR tabanlı SL/TP)
#   - BE stop (1.5R), time stop, cooldown
#   - ADX / Rejim / Trend filtreleri + confirm_bars
#   - Detaylı kayıt + Excel/CSV export
# ==============================================================
class MlStrategy(bt.Strategy):
    PROGRESS_CSV = "opt_progress.csv"

    params = (
        # --- Trade/Sinyal ---
        ("trade_fast_sma", 10),
        ("trade_slow_sma", 50),
        ("trend_fast_sma", 10),
        ("trend_slow_sma", 50),
        ("allow_short", True),
        ("adx_period", 14),
        ("adx_min", 18),                 # gevşetildi
        ("min_cross_strength", 0.35),    # gevşetildi
        ("confirm_bars", 1),             # sinyal teyidi
        ("cooldown_bars", 24),           # ~2 saat (5m)
        ("time_stop_bars", 120),         # ~10 saat (5m)

        # --- Entry ayarları ---
        ("entry_atr_mult", 0.35),        # stop-entry uzaklığı
        ("swing_lookback", 9),           # swing HL aralığı
        ("entry_timeout_bars", 18),      # ~1.5 saat
        ("trade_start_hour", 6),         # UTC
        ("trade_end_hour", 22),

        # --- Stops ---
        ("use_dynamic_risk", True),   # OTR haritası aktif
        ("use_atr_stops", True),
        ("atr_period", 14),
        ("atr_sl_mult", 2.0),
        ("atr_tp_mult", 3.5),
        ("stop_loss", 0.018),
        ("take_profit", 0.055),
        ("regime_sl_tp", dict()),     # {regime: (sl_mult, tp_mult)}

        # --- Rejim ---
        ("good_regime_id", None),     # K-means auto select geldiyse dolu olur

        # --- ARIMA filtresi ---
        ("arima_enabled", False),
        ("arima_order", (1, 1, 0)),
        ("arima_lookback", 50),
        ("arima_forecast_steps", 3),

        # --- Meta-labeler & bahis ---
        ("use_meta", False),
        ("meta_threshold", 0.55),
        ("meta_model", None),
        ("bet_sizing", False),
        ("max_pos_pct", 0.30),
        ("fallback_bet", 0.20),

        # --- Diğer ---
        ("printlog", False),
    )

    def __init__(self):
        self.do_log = not self.cerebro.p.optreturn

        # Veri referansları
        self.data_trade = self.datas[0]   # 5m
        self.data_trend = self.datas[1]   # 1h (rejim)
        self.current_regime = self.data_trend.lines.regime if hasattr(self.data_trend.lines, "regime") else None

        # Emir/state
        self.order = None                 # aktif emir (entry/exit/SL/TP)
        self.entry_order = None           # bekleyen stop-entry
        self.sl_order = None
        self.tp_order = None
        self.bar_opened = None
        self.entry_price = None
        self.stop_price = None
        self._risk = None
        self.be_stop_active = False
        self.last_trade_bar = -10**9
        self.entry_size_signed = None
        self.last_exit_tag = None
        self.pending_entry = False        # stop-entry bekleniyor mu?

        # İndikatörler
        self.trade_fast = SafeSMA(self.data_trade, period=int(self.p.trade_fast_sma))
        self.trade_slow = SafeSMA(self.data_trade, period=int(self.p.trade_slow_sma))
        self.trade_crossover = btind.CrossOver(self.trade_fast, self.trade_slow)
        self.trend_fast = SafeSMA(self.data_trend, period=int(self.p.trend_fast_sma))
        self.trend_slow = SafeSMA(self.data_trend, period=int(self.p.trend_slow_sma))
        self.adx = btind.AverageDirectionalMovementIndex(self.data_trade, period=int(self.p.adx_period))
        self.atr = btind.ATR(self.data_trade, period=int(self.p.atr_period))
        self.trade_ret = bt.indicators.PercentChange(self.data_trade.close, period=1)
        self.trade_rv30 = bt.indicators.StandardDeviation(self.trade_ret, period=30) * (288 * 252) ** 0.5

        # Kayıt yapıları
        self.trades = []                 # ayrıntılı trade kayıtları (Excel için)
        self.closed_trades_pnl = []      # Monte Carlo için
        self.trade_history = []          # zoom-in grafikler için
        self.plot_events = []            # grafik işaretleri
        self.equity_curve = []           # her barda broker value
        self.equity_times = []

        # Param snapshot
        self.run_params = {}

        if self.do_log and self.p.printlog and self.p.use_dynamic_risk:
            print(f"--- OTR aktif. Harita: {self.p.regime_sl_tp}")

    # ----------------------------------------------------------
    # Yardımcılar
    def log(self, txt, dt=None, doprint=False):
        if self.do_log and (self.p.printlog or doprint):
            dt = dt or self.data_trade.datetime.datetime(0)
            print(f"{dt.isoformat()}, {txt}")

    def _lt_sl_tp(self, price, side):
        # OTR
        if self.p.use_dynamic_risk and self.current_regime is not None:
            try:
                reg = int(self.current_regime[0])
                mult = self.p.regime_sl_tp.get(reg)
                if mult:
                    sl_m, tp_m = mult
                    atr = float(self.atr[0])
                    sl_off, tp_off = sl_m * atr, tp_m * atr
                    if side == "long":
                        return price - sl_off, price + tp_off
                    else:
                        return price + sl_off, price - tp_off
            except Exception as e:
                self.log(f"OTR hata/eksik: {e}")
        # ATR fallback
        if self.p.use_atr_stops and len(self.atr) >= int(self.p.atr_period):
            atr = float(self.atr[0])
            sl_off = self.p.atr_sl_mult * atr
            tp_off = self.p.atr_tp_mult * atr
            if side == "long":
                return price - sl_off, price + tp_off
            else:
                return price + sl_off, price - tp_off
        # Yüzdesel fallback
        if side == "long":
            return price * (1.0 - self.p.stop_loss), price * (1.0 + self.p.take_profit)
        else:
            return price * (1.0 + self.p.stop_loss), price * (1.0 - self.p.take_profit)

    def get_arima_forecast(self) -> bool:
        try:
            if not self.p.arima_enabled:
                return True
            look = int(self.p.arima_lookback)
            steps = int(self.p.arima_forecast_steps)
            arr = list(self.data_trade.close.get(size=look))
            if len(arr) < look:
                return False
            model = ARIMA(arr, order=self.p.arima_order)
            fit = model.fit()
            fc = fit.forecast(steps=steps)
            return float(fc[-1]) > float(arr[-1])
        except Exception as e:
            self.log(f"ARIMA hata: {e}")
            return False

    def _confirm_cross(self) -> int:
        """
        confirm_bars > 0 ise son N barın tamamında cross yönü aynı mı?
        >0: bull, <0: bear, 0: yok
        """
        n = max(1, int(self.p.confirm_bars))
        val = 0
        for i in range(n):
            cv = int(np.sign(float(self.trade_fast[-i] - self.trade_slow[-i])))
            # 0 (eşit) durumda teyit bozulur
            if cv == 0:
                return 0
            if i == 0:
                val = cv
            elif cv != val:
                return 0
        return val

    def _swing_levels(self):
        lb = int(self.p.swing_lookback)
        highs = [float(self.data_trade.high[-i]) for i in range(lb)]
        lows = [float(self.data_trade.low[-i]) for i in range(lb)]
        return max(highs), min(lows)

    # ----------------------------------------------------------
    def start(self):
        # Param snapshot/CSV log
        self.run_params = dict(
            fast=self.p.trade_fast_sma,
            slow=self.p.trade_slow_sma,
            trend_fast=self.p.trend_fast_sma,
            trend_slow=self.p.trend_slow_sma,
            adx_min=self.p.adx_min,
            min_cross_strength=self.p.min_cross_strength,
            max_pos_pct=self.p.max_pos_pct,
            dynamic_risk=self.p.use_dynamic_risk,
            atr_period=self.p.atr_period,
            regime_map=str(self.p.regime_sl_tp),
            good_regime=self.p.good_regime_id,
            allow_short=self.p.allow_short,
        )
        try:
            need_header = not os.path.exists(self.PROGRESS_CSV) or os.path.getsize(self.PROGRESS_CSV) == 0
            with open(self.PROGRESS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(["event", "fast", "slow", "datetime", "equity"])
                w.writerow(["START", int(self.p.trade_fast_sma), int(self.p.trade_slow_sma),
                            self.data_trade.datetime.datetime(0).isoformat(), f"{self.broker.getvalue():.2f}"])
        except Exception:
            pass

    def stop(self):
        try:
            with open(self.PROGRESS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(["STOP", int(self.p.trade_fast_sma), int(self.p.trade_slow_sma),
                            self.data_trade.datetime.datetime(0).isoformat(), f"{self.broker.getvalue():.2f}"])
        except Exception:
            pass
        # Export
        try:
            self._export_trades_excel()
        except Exception as e:
            self.log(f"Excel export başarısız: {e}", doprint=True)

    # ----------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            dt = self.data_trade.datetime.datetime(0)
            px = order.executed.price
            side = "BUY" if order.isbuy() else "SELL"
            tag = order.info.get("tag", "")

            # Grafik event kaydı
            if tag in ("ENTRY", "ENTRY_SHORT", "SL", "TP", "TIME_STOP", "EXIT", "EXIT_SHORT"):
                self.plot_events.append((dt, px, side, tag))

            # LONG giriş tamam
            if order.isbuy() and tag == "ENTRY":
                self.pending_entry = False
                self.entry_order = None
                sl_p, tp_p = self._lt_sl_tp(px, "long")
                self.sl_order = self.sell(data=self.data_trade, exectype=bt.Order.Stop, price=sl_p)
                self.sl_order.addinfo(tag="SL")
                self.tp_order = self.sell(data=self.data_trade, exectype=bt.Order.Limit, price=tp_p)
                self.tp_order.addinfo(tag="TP")
                self.entry_price = px
                self.stop_price = sl_p
                self._risk = self.entry_price - self.stop_price
                self.bar_opened = len(self.data_trade)
                self.be_stop_active = False
                self.last_trade_bar = len(self.data_trade)   # zaman aşımı sayacı
                try:
                    self.entry_size_signed = float(self.position.size)
                except Exception:
                    self.entry_size_signed = float(order.executed.size)

            # SHORT giriş tamam
            elif order.issell() and tag == "ENTRY_SHORT":
                self.pending_entry = False
                self.entry_order = None
                sl_p, tp_p = self._lt_sl_tp(px, "short")
                self.sl_order = self.buy(data=self.data_trade, exectype=bt.Order.Stop, price=sl_p)
                self.sl_order.addinfo(tag="SL")
                self.tp_order = self.buy(data=self.data_trade, exectype=bt.Order.Limit, price=tp_p)
                self.tp_order.addinfo(tag="TP")
                self.entry_price = px
                self.stop_price = sl_p
                self._risk = abs(self.entry_price - self.stop_price)
                self.bar_opened = len(self.data_trade)
                self.be_stop_active = False
                self.last_trade_bar = len(self.data_trade)
                try:
                    self.entry_size_signed = float(self.position.size)
                except Exception:
                    self.entry_size_signed = float(order.executed.size)

            # Çıkış emirleriyle pozisyon kapanmış olabilir
            if tag in ("SL", "TP", "TIME_STOP", "EXIT", "EXIT_SHORT"):
                self.last_exit_tag = tag

            # Pozisyon kapandıysa state temizliği
            if self.position.size == 0 and tag in ("SL", "TP", "TIME_STOP", "EXIT", "EXIT_SHORT"):
                self.bar_opened = None
                self.entry_price = None
                self._risk = None
                self.be_stop_active = False
                self.entry_size_signed = None

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.printlog:
                self.log(f"Emir iptal/red/marjin: {order.info.get('tag','')} -> {order.getstatusname()}")
            if self.order is order:
                self.order = None
            if self.entry_order is order:
                self.pending_entry = False
                self.entry_order = None

    # ----------------------------------------------------------
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        pnl = float(trade.pnlcomm)
        bars_held = (len(self.data_trade) - self.bar_opened) if self.bar_opened is not None else None

        # Kapanış fiyatını güvenli yöntemle tahmin et (sizestart yok)
        try:
            signed_size = self.entry_size_signed
            if signed_size in (None, 0):
                close_px = trade.price
            else:
                close_px = trade.price + (trade.pnl / signed_size)
        except Exception:
            close_px = trade.price

        entry_dt = bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(trade.dtclose)
        entry_px_final = float(self.entry_price) if self.entry_price else float(trade.price)

        # girişteki rejim snapshot
        try:
            entry_regime = int(self.current_regime[0]) if self.current_regime is not None else None
        except Exception:
            entry_regime = None

        # Excel için detay
        self.trades.append(dict(
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            side="LONG" if trade.long else "SHORT",
            entry_px=entry_px_final,
            exit_px=float(close_px),
            pnl=pnl,
            bars=bars_held,
            entry_adx=float(self.adx[0]) if len(self.adx) else np.nan,
            entry_atr=float(self.atr[0]) if len(self.atr) else np.nan,
            regime=entry_regime,
            exit_reason=self.last_exit_tag,
        ))
        # Monte Carlo
        self.closed_trades_pnl.append(pnl)
        # Zoom-in
        self.trade_history.append({
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "entry_px": entry_px_final,
            "exit_px": float(close_px),
            "pnl": pnl,
            "is_long": bool(trade.long),
        })

        # state reset
        self.last_trade_bar = len(self.data_trade)
        self.order = None
        self.sl_order = None
        self.tp_order = None
        self.bar_opened = None
        self.entry_price = None
        self._risk = None
        self.be_stop_active = False
        self.entry_size_signed = None
        self.last_exit_tag = None

    # ----------------------------------------------------------
    def next(self):
        # Equity curve
        self.equity_curve.append(float(self.broker.getvalue()))
        self.equity_times.append(self.data_trade.datetime.datetime(0))

        # Warmups
        if (len(self.trend_slow) < int(self.p.trend_slow_sma) or
            len(self.trade_slow) < int(self.p.trade_slow_sma) or
            len(self.adx) < int(self.p.adx_period) or
            len(self.atr) < int(self.p.atr_period) or
            len(self.trade_rv30) < 30):
            return

        # Saat filtresi
        dt_now = self.data_trade.datetime.datetime(0)
        if not (int(self.p.trade_start_hour) <= dt_now.hour <= int(self.p.trade_end_hour)):
            return

        # açık emir varsa (SL/TP/exit) — ama entry emirleri için ayrıca kontrol var
        if self.order and (self.order is not self.entry_order):
            return

        # pending stop-entry varsa zaman aşımı / market fallback
        if self.pending_entry and (self.entry_order is not None):
            elapsed = len(self.data_trade) - self.last_trade_bar
            if elapsed >= int(self.p.entry_timeout_bars):
                # sinyal hâlâ aynı yönde mi?
                cross_val = self._confirm_cross()
                is_long_trend = self.trend_fast[0] > self.trend_slow[0]
                reg_ok = True
                if self.p.good_regime_id is not None and self.current_regime is not None:
                    reg_ok = (self.current_regime[0] == self.p.good_regime_id)

                try:
                    self.broker.cancel(self.entry_order)
                except Exception:
                    pass
                self.entry_order = None
                self.pending_entry = False

                if cross_val > 0 and is_long_trend and reg_ok:
                    cash_to_use = self.broker.getvalue() * (float(self.p.max_pos_pct) * float(self.p.fallback_bet))
                    px = float(self.data_trade.close[0])
                    size = cash_to_use / max(px, 1e-9)
                    self.order = self.buy(data=self.data_trade, size=size)
                    self.order.addinfo(tag="ENTRY")
                    return

                if self.p.allow_short and cross_val < 0 and (not is_long_trend) and reg_ok:
                    cash_to_use = self.broker.getvalue() * (float(self.p.max_pos_pct) * float(self.p.fallback_bet))
                    px = float(self.data_trade.close[0])
                    size = cash_to_use / max(px, 1e-9)
                    self.order = self.sell(data=self.data_trade, size=size)
                    self.order.addinfo(tag="ENTRY_SHORT")
                    return
            else:
                # bekleme sürüyor
                return

        # Pozisyon yönetimi
        if self.position:
            cp = float(self.data_trade.close[0])

            # time stop
            if self.bar_opened is not None:
                held = len(self.data_trade) - self.bar_opened
                if held >= int(self.p.time_stop_bars):
                    self.log("TIME STOP")
                    self.order = self.close(data=self.data_trade)
                    self.order.addinfo(tag="TIME_STOP")
                    return

            # BE stop
            if self._risk is not None and self.entry_price is not None and not self.be_stop_active:
                if self.position.size > 0 and cp >= self.entry_price + 1.5 * self._risk:
                    be = self.entry_price
                    if (self.sl_order is None) or (self.sl_order.price < be):
                        try:
                            if self.sl_order:
                                self.broker.cancel(self.sl_order)
                        except Exception:
                            pass
                        self.sl_order = self.sell(data=self.data_trade, exectype=bt.Order.Stop, price=be)
                        self.sl_order.addinfo(tag="SL")
                        self.be_stop_active = True
                elif self.position.size < 0 and cp <= self.entry_price - 1.5 * self._risk:
                    be = self.entry_price
                    if (self.sl_order is None) or (self.sl_order.price > be):
                        try:
                            if self.sl_order:
                                self.broker.cancel(self.sl_order)
                        except Exception:
                            pass
                        self.sl_order = self.buy(data=self.data_trade, exectype=bt.Order.Stop, price=be)
                        self.sl_order.addinfo(tag="SL")
                        self.be_stop_active = True

            # Çıkış mantığı
            try:
                is_long_trend = self.trend_fast[0] > self.trend_slow[0]
                cross_val = self.trade_crossover[0]
            except IndexError:
                return

            if self.position.size > 0:
                weak = (self.adx.lines.adx[0] < float(self.p.adx_min))
                bad = (self.p.good_regime_id is not None and self.current_regime is not None and self.current_regime[0] != self.p.good_regime_id)
                if ((cross_val < 0) and (weak or bad)) or (not is_long_trend):
                    self.order = self.close(data=self.data_trade)
                    self.order.addinfo(tag="EXIT")
                    return
            else:  # short
                weak = (self.adx.lines.adx[0] < float(self.p.adx_min))
                bad = (self.p.good_regime_id is not None and self.current_regime is not None and self.current_regime[0] != self.p.good_regime_id)
                if ((cross_val > 0) and (weak or bad)) or is_long_trend:
                    self.order = self.close(data=self.data_trade)
                    self.order.addinfo(tag="EXIT_SHORT")
                    return
            return

        # Cooldown
        if (len(self.data_trade) - self.last_trade_bar) < int(self.p.cooldown_bars):
            return

        # Giriş kontrolleri
        try:
            if self.adx.lines.adx[0] < float(self.p.adx_min):
                return
            is_long_trend = self.trend_fast[0] > self.trend_slow[0]
            is_short_trend = self.trend_fast[0] < self.trend_slow[0]
            reg_ok = True
            if self.p.good_regime_id is not None and self.current_regime is not None:
                reg_ok = (self.current_regime[0] == self.p.good_regime_id)

            # onaylı cross
            cross_dir = self._confirm_cross()
            atr_now = float(self.atr[0])
            if atr_now < 1e-9:
                return
            cross_strength = abs(float(self.trade_fast[0] - self.trade_slow[0])) / atr_now
            if cross_strength < float(self.p.min_cross_strength):
                return
        except IndexError:
            return

        # Stop-entry seviyeleri
        swing_h, swing_l = self._swing_levels()
        entry_off = float(self.p.entry_atr_mult) * float(self.atr[0])

        # LONG sinyal: stop-buy
        if (cross_dir > 0) and is_long_trend and reg_ok and not self.pending_entry:
            if not self.get_arima_forecast():
                return
            proba = 1.0
            if self.p.use_meta and self.p.meta_model is not None:
                x = self._get_current_features()
                proba = self.p.meta_model.proba(x)
                if proba < float(self.p.meta_threshold):
                    return
            bet = self.p.fallback_bet
            if self.p.bet_sizing and self.p.meta_model is not None:
                bet = max(0.0, min(1.0, self.p.meta_model.bet_size(proba)))
                if bet <= 0:
                    return

            cash_to_use = self.broker.getvalue() * (float(self.p.max_pos_pct) * float(bet))
            px = float(self.data_trade.close[0])
            size = cash_to_use / max(px, 1e-9)
            stop_px = max(swing_h, px) + entry_off
            self.entry_order = self.buy(data=self.data_trade, size=size, exectype=bt.Order.Stop, price=stop_px)
            self.entry_order.addinfo(tag="ENTRY")
            self.pending_entry = True
            self.last_trade_bar = len(self.data_trade)
            return

        # SHORT sinyal: stop-sell
        if self.p.allow_short and (cross_dir < 0) and is_short_trend and reg_ok and not self.pending_entry:
            if not self.get_arima_forecast():
                # arima_enabled ise False dönerse short açmak için tersini kullanıyorduk;
                # v38: ARIMA filtreyi "True ise devam" şeklinde tek kapı olarak tutuyoruz.
                return
            proba = 1.0
            if self.p.use_meta and self.p.meta_model is not None:
                x = self._get_current_features()
                long_p = self.p.meta_model.proba(x)
                proba = 1.0 - long_p
                if long_p > (1.0 - self.p.meta_threshold):
                    return
            bet = self.p.fallback_bet
            if self.p.bet_sizing and self.p.meta_model is not None:
                scaled = (proba - self.p.meta_threshold) / max(1e-9, (1.0 - self.p.meta_threshold))
                bet = max(0.0, min(1.0, 2 * scaled - 1))
                if bet <= 0:
                    return

            cash_to_use = self.broker.getvalue() * (float(self.p.max_pos_pct) * float(bet))
            px = float(self.data_trade.close[0])
            size = cash_to_use / max(px, 1e-9)
            stop_px = min(swing_l, px) - entry_off
            self.entry_order = self.sell(data=self.data_trade, size=size, exectype=bt.Order.Stop, price=stop_px)
            self.entry_order.addinfo(tag="ENTRY_SHORT")
            self.pending_entry = True
            self.last_trade_bar = len(self.data_trade)
            return

    # ----------------------------------------------------------
    def _get_current_features(self) -> pd.Series:
        L = len(self.data_trade)
        C = self.data_trade.close
        feat = {
            "ret1": C[0] / C[-1] - 1 if L > 1 else 0.0,
            "ret5": C[0] / C[-5] - 1 if L > 5 else 0.0,
            "ret10": C[0] / C[-10] - 1 if L > 10 else 0.0,
            "rv_30": float(self.trade_rv30[0]) if len(self.trade_rv30) else 0.0,
            "atr_14": float(self.atr[0]) if len(self.atr) else 0.0,
            "roc_60": C[0] / C[-60] - 1 if L > 60 else 0.0,
            "fd_d05": 0.0,
            "lz": 0.5,
            "regime": float(self.current_regime[0]) if (self.current_regime is not None and len(self.current_regime)) else 0.0,
        }
        return pd.Series(feat)

    # ----------------------------------------------------------
    def _export_trades_excel(self):
        # Trades DataFrame
        df = pd.DataFrame(self.trades)
        # Equity ve MaxDD
        eq = pd.Series(self.equity_curve, index=pd.to_datetime(self.equity_times))
        if not eq.empty:
            rolling_max = eq.cummax()
            dd = (eq - rolling_max) / rolling_max
            maxdd = dd.min() if len(dd) else np.nan
        else:
            maxdd = np.nan

        # CSV yedek
        if len(df):
            df.to_csv("trades.csv", index=False, encoding="utf-8")
        else:
            # boş da olsa header yazalım
            pd.DataFrame(columns=["entry_dt","exit_dt","side","entry_px","exit_px","pnl","bars","entry_adx","entry_atr","regime","exit_reason"]).to_csv("trades.csv", index=False)

        # Excel
        with pd.ExcelWriter("trades.xlsx", engine="xlsxwriter") as writer:
            # Parameters
            pd.DataFrame([self.run_params]).to_excel(writer, index=False, sheet_name="Parameters")

            # Summary
            summary = {
                "start_cash": getattr(self.broker, "startingcash", np.nan),
                "end_value": float(self.broker.getvalue()),
                "net_pl": float(self.broker.getvalue()) - float(getattr(self.broker, "startingcash", float("nan"))),
                "trades": len(df),
                "wins": int((df["pnl"] > 0).sum()) if len(df) else 0,
                "losses": int((df["pnl"] <= 0).sum()) if len(df) else 0,
                "win_rate": float((df["pnl"] > 0).mean()) if len(df) else np.nan,
                "avg_win": float(df.loc[df["pnl"] > 0, "pnl"].mean()) if len(df) and (df["pnl"] > 0).any() else np.nan,
                "avg_loss": float(df.loc[df["pnl"] <= 0, "pnl"].mean()) if len(df) and (df["pnl"] <= 0).any() else np.nan,
                "max_dd_pct": float(maxdd) if pd.notna(maxdd) else np.nan,
            }
            pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="Summary")

            # Trades
            df.to_excel(writer, index=False, sheet_name="Trades")

            # Last10
            df.tail(10).to_excel(writer, index=False, sheet_name="Last10")

            # Basit formatlama
            wb = writer.book
            pct = wb.add_format({"num_format": "0.00%"})
            money = wb.add_format({"num_format": "0.00"})
            try:
                ws = writer.sheets["Summary"]
                ws.set_column("A:A", 18)
                ws.set_column("B:C", 14, money)
                ws.set_column("G:G", 10)
                ws.set_column("H:I", 12, money)
                ws.set_column("J:J", 12, pct)
                ws.set_column("K:K", 12, money)
            except Exception:
                pass

        self.log("Excel export: trades.xlsx ve trades.csv oluşturuldu", doprint=True)
