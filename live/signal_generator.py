# live/signal_generator.py
"""
Saf pandas/numpy sinyal üreteci — Backtrader bağımlılığı yok.
strategies/ml_strategy.py (v39) next() mantığını portlar.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass


@dataclass
class Signal:
    direction: str | None   # "long", "short", None
    price: float            # sinyal anındaki fiyat
    sl_price: float = 0.0
    tp_price: float = 0.0
    strength: float = 0.0
    reason: str = ""


class SignalGenerator:
    def __init__(
        self,
        trade_fast_sma: int = 10,
        trade_slow_sma: int = 50,
        trend_fast_sma: int = 10,
        trend_slow_sma: int = 50,
        adx_period: int = 14,
        adx_min: float = 20,
        atr_period: int = 14,
        min_cross_strength: float = 0.25,
        cooldown_bars: int = 12,
        confirm_bars: int = 1,
        good_regime_id: int | None = None,
        filter_trading_hours: bool = True,
        trade_start_hour: int = 6,
        trade_end_hour: int = 22,
        # SL/TP
        use_dynamic_risk: bool = True,
        regime_sl_tp: dict | None = None,
        use_atr_stops: bool = True,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.5,
        stop_loss: float = 0.018,
        take_profit: float = 0.055,
        allow_short: bool = True,
        # Zaman stop
        time_stop_bars: int = 120,
    ):
        self.trade_fast_sma = trade_fast_sma
        self.trade_slow_sma = trade_slow_sma
        self.trend_fast_sma = trend_fast_sma
        self.trend_slow_sma = trend_slow_sma
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.atr_period = atr_period
        self.min_cross_strength = min_cross_strength
        self.cooldown_bars = cooldown_bars
        self.confirm_bars = confirm_bars
        self.good_regime_id = good_regime_id
        self.filter_trading_hours = filter_trading_hours
        self.trade_start_hour = trade_start_hour
        self.trade_end_hour = trade_end_hour
        self.use_dynamic_risk = use_dynamic_risk
        self.regime_sl_tp = regime_sl_tp or {}
        self.use_atr_stops = use_atr_stops
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allow_short = allow_short
        self.time_stop_bars = time_stop_bars

    def compute_indicators(
        self, df_trade: pd.DataFrame, df_trend: pd.DataFrame
    ) -> dict:
        """Trade ve trend verisi üzerinde indikatörleri hesapla."""
        ind = {}

        # Trade timeframe SMA'lar
        ind["trade_sma_fast"] = ta.sma(df_trade["close"], length=self.trade_fast_sma)
        ind["trade_sma_slow"] = ta.sma(df_trade["close"], length=self.trade_slow_sma)

        # Trend timeframe SMA'lar
        ind["trend_sma_fast"] = ta.sma(df_trend["close"], length=self.trend_fast_sma)
        ind["trend_sma_slow"] = ta.sma(df_trend["close"], length=self.trend_slow_sma)

        # ADX
        adx_df = ta.adx(
            df_trade["high"], df_trade["low"], df_trade["close"],
            length=self.adx_period,
        )
        if adx_df is not None:
            adx_col = f"ADX_{self.adx_period}"
            ind["adx"] = adx_df[adx_col] if adx_col in adx_df.columns else adx_df.iloc[:, 0]
        else:
            ind["adx"] = pd.Series(np.nan, index=df_trade.index)

        # ATR
        ind["atr"] = ta.atr(
            df_trade["high"], df_trade["low"], df_trade["close"],
            length=self.atr_period,
        )

        ind["trade_close"] = df_trade["close"]
        ind["trade_index"] = df_trade.index
        ind["trend_index"] = df_trend.index

        return ind

    def _calc_sl_tp(self, price: float, side: str, atr_val: float, regime: int | None):
        """SL/TP hesapla — OTR → ATR → yüzdesel fallback."""
        # OTR (rejime göre)
        if self.use_dynamic_risk and regime is not None:
            mult = self.regime_sl_tp.get(regime)
            if mult:
                sl_m, tp_m = mult
                sl_off, tp_off = sl_m * atr_val, tp_m * atr_val
                if side == "long":
                    return price - sl_off, price + tp_off
                else:
                    return price + sl_off, price - tp_off

        # ATR fallback
        if self.use_atr_stops and atr_val > 0:
            sl_off = self.atr_sl_mult * atr_val
            tp_off = self.atr_tp_mult * atr_val
            if side == "long":
                return price - sl_off, price + tp_off
            else:
                return price + sl_off, price - tp_off

        # Yüzdesel fallback
        if side == "long":
            return price * (1 - self.stop_loss), price * (1 + self.take_profit)
        else:
            return price * (1 + self.stop_loss), price * (1 - self.take_profit)

    def _confirm_cross(self, sma_fast: pd.Series, sma_slow: pd.Series) -> int:
        """Son confirm_bars barının tamamında cross yönü aynı mı? +1/-1/0"""
        n = max(1, self.confirm_bars)
        if len(sma_fast) < n or len(sma_slow) < n:
            return 0

        val = 0
        for i in range(n):
            idx = -(i + 1)
            diff = float(sma_fast.iloc[idx]) - float(sma_slow.iloc[idx])
            cv = int(np.sign(diff))
            if cv == 0:
                return 0
            if i == 0:
                val = cv
            elif cv != val:
                return 0
        return val

    def get_signal(
        self,
        indicators: dict,
        current_regime: int | None,
        bars_since_last_trade: int,
        current_dt: pd.Timestamp | None = None,
    ) -> Signal:
        """Son bar üzerinde giriş sinyali üret."""
        close = indicators["trade_close"]
        if close.empty:
            return Signal(direction=None, price=0.0, reason="veri yok")

        price = float(close.iloc[-1])

        # Saat filtresi
        if self.filter_trading_hours and current_dt is not None:
            hour = current_dt.hour
            if not (self.trade_start_hour <= hour <= self.trade_end_hour):
                return Signal(direction=None, price=price, reason="saat dışı")

        # Cooldown
        if bars_since_last_trade < self.cooldown_bars:
            return Signal(direction=None, price=price, reason="cooldown")

        sma_fast = indicators["trade_sma_fast"]
        sma_slow = indicators["trade_sma_slow"]
        adx = indicators["adx"]
        atr = indicators["atr"]
        trend_fast = indicators["trend_sma_fast"]
        trend_slow = indicators["trend_sma_slow"]

        # Warmup
        min_len = max(self.trade_slow_sma, self.trend_slow_sma, self.adx_period, self.atr_period) + 5
        if len(sma_fast.dropna()) < 2 or len(sma_slow.dropna()) < 2:
            return Signal(direction=None, price=price, reason="warmup")

        # Son değerler
        adx_val = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0
        atr_val = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0

        # ADX filtresi
        if adx_val < self.adx_min:
            return Signal(direction=None, price=price, reason=f"ADX düşük ({adx_val:.1f})")

        # Trend yönü
        if pd.isna(trend_fast.iloc[-1]) or pd.isna(trend_slow.iloc[-1]):
            return Signal(direction=None, price=price, reason="trend warmup")
        is_long_trend = float(trend_fast.iloc[-1]) > float(trend_slow.iloc[-1])
        is_short_trend = float(trend_fast.iloc[-1]) < float(trend_slow.iloc[-1])

        # Rejim filtresi
        reg_ok = True
        if self.good_regime_id is not None and current_regime is not None:
            reg_ok = (current_regime == self.good_regime_id)

        if not reg_ok:
            return Signal(direction=None, price=price, reason="kötü rejim")

        # Cross teyidi
        cross_dir = self._confirm_cross(sma_fast, sma_slow)

        # Cross gücü
        if atr_val < 1e-9:
            return Signal(direction=None, price=price, reason="ATR=0")
        cross_str = abs(float(sma_fast.iloc[-1]) - float(sma_slow.iloc[-1])) / atr_val
        if cross_str < self.min_cross_strength:
            return Signal(direction=None, price=price, reason=f"cross zayıf ({cross_str:.3f})")

        # LONG sinyal
        if cross_dir > 0 and is_long_trend and reg_ok:
            sl, tp = self._calc_sl_tp(price, "long", atr_val, current_regime)
            return Signal(
                direction="long", price=price,
                sl_price=sl, tp_price=tp,
                strength=cross_str, reason="SMA cross UP + trend OK",
            )

        # SHORT sinyal
        if self.allow_short and cross_dir < 0 and is_short_trend and reg_ok:
            sl, tp = self._calc_sl_tp(price, "short", atr_val, current_regime)
            return Signal(
                direction="short", price=price,
                sl_price=sl, tp_price=tp,
                strength=cross_str, reason="SMA cross DOWN + trend OK",
            )

        return Signal(direction=None, price=price, reason="sinyal yok")

    def should_exit(
        self,
        indicators: dict,
        position_side: str,
        bars_held: int,
        current_regime: int | None,
    ) -> str | None:
        """Açık pozisyon için çıkış sinyali. None=tut, str=çıkış nedeni."""
        # Zaman stop
        if bars_held >= self.time_stop_bars:
            return "TIME_STOP"

        sma_fast = indicators["trade_sma_fast"]
        sma_slow = indicators["trade_sma_slow"]
        trend_fast = indicators["trend_sma_fast"]
        trend_slow = indicators["trend_sma_slow"]
        adx = indicators["adx"]

        if sma_fast.empty or sma_slow.empty:
            return None

        # Cross yönü
        fast_val = float(sma_fast.iloc[-1]) if pd.notna(sma_fast.iloc[-1]) else None
        slow_val = float(sma_slow.iloc[-1]) if pd.notna(sma_slow.iloc[-1]) else None
        if fast_val is None or slow_val is None:
            return None
        cross_val = 1 if fast_val > slow_val else -1

        adx_val = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0
        weak = adx_val < self.adx_min

        bad_regime = False
        if self.good_regime_id is not None and current_regime is not None:
            bad_regime = (current_regime != self.good_regime_id)

        # Trend yönü
        if pd.notna(trend_fast.iloc[-1]) and pd.notna(trend_slow.iloc[-1]):
            is_long_trend = float(trend_fast.iloc[-1]) > float(trend_slow.iloc[-1])
        else:
            is_long_trend = True  # varsayılan

        if position_side == "long":
            if (cross_val < 0 and (weak or bad_regime)) or not is_long_trend:
                return "EXIT"
        elif position_side == "short":
            if (cross_val > 0 and (weak or bad_regime)) or is_long_trend:
                return "EXIT_SHORT"

        return None
