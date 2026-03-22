# strategies/rsi_bollinger.py
"""
RSI + Bollinger Bands stratejisi.

Mantık:
- RSI oversold + fiyat alt Bollinger bandına dokundu → LONG
- RSI overbought + fiyat üst Bollinger bandına dokundu → SHORT
- SL/TP: ATR bazlı (broker stop/limit order)
- Volume filtresi: ortalamanın üstünde volume gerekli
"""
import backtrader as bt


class RsiBollingerStrategy(bt.Strategy):
    params = dict(
        # RSI
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        # Bollinger
        bb_period=20,
        bb_devfactor=2.0,
        # Volume
        vol_period=20,
        vol_mult=1.0,        # volume > vol_mult * ortalama
        # Risk
        atr_period=14,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        stop_loss=0.02,      # yüzdesel fallback
        take_profit=0.04,
        # Diğer
        allow_short=True,
        cooldown_bars=10,
        printlog=False,
    )

    def __init__(self):
        self.data_close = self.datas[0].close

        # İndikatörler
        self.rsi = bt.indicators.RSI(self.data_close, period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(
            self.data_close,
            period=self.p.bb_period,
            devfactor=self.p.bb_devfactor,
        )
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)
        self.vol_sma = bt.indicators.SMA(self.datas[0].volume, period=self.p.vol_period)

        # Emir state
        self.order = None
        self.sl_order = None
        self.tp_order = None
        self.entry_price = None
        self.last_trade_bar = -10**9

    def log(self, txt):
        if self.p.printlog:
            dt = self.datas[0].datetime.datetime(0)
            print(f"{dt.isoformat()} - {txt}")

    def _cancel_bracket(self):
        for o in (self.sl_order, self.tp_order):
            if o is not None:
                try:
                    self.broker.cancel(o)
                except Exception:
                    pass
        self.sl_order = None
        self.tp_order = None

    def _calc_sl_tp(self, price, side):
        """ATR bazlı SL/TP, yüzdesel fallback."""
        atr_val = float(self.atr[0]) if len(self.atr) >= self.p.atr_period else 0
        if atr_val > 0:
            sl_off = self.p.atr_sl_mult * atr_val
            tp_off = self.p.atr_tp_mult * atr_val
        else:
            sl_off = price * self.p.stop_loss
            tp_off = price * self.p.take_profit

        if side == "long":
            return price - sl_off, price + tp_off
        else:
            return price + sl_off, price - tp_off

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            tag = order.info.get("tag", "")

            if order.isbuy() and tag == "ENTRY":
                px = order.executed.price
                self.entry_price = px
                sl, tp = self._calc_sl_tp(px, "long")
                self.sl_order = self.sell(exectype=bt.Order.Stop, price=sl)
                self.sl_order.addinfo(tag="SL")
                self.tp_order = self.sell(exectype=bt.Order.Limit, price=tp)
                self.tp_order.addinfo(tag="TP")
                self.log(f"LONG ENTRY @ {px:.2f}, SL={sl:.2f}, TP={tp:.2f}")

            elif order.issell() and tag == "ENTRY_SHORT":
                px = order.executed.price
                self.entry_price = px
                sl, tp = self._calc_sl_tp(px, "short")
                self.sl_order = self.buy(exectype=bt.Order.Stop, price=sl)
                self.sl_order.addinfo(tag="SL")
                self.tp_order = self.buy(exectype=bt.Order.Limit, price=tp)
                self.tp_order.addinfo(tag="TP")
                self.log(f"SHORT ENTRY @ {px:.2f}, SL={sl:.2f}, TP={tp:.2f}")

            elif tag in ("SL", "TP", "EXIT"):
                px = order.executed.price
                self.log(f"{tag} @ {px:.2f}")
                if tag == "SL" and self.tp_order:
                    try:
                        self.broker.cancel(self.tp_order)
                    except Exception:
                        pass
                elif tag == "TP" and self.sl_order:
                    try:
                        self.broker.cancel(self.sl_order)
                    except Exception:
                        pass
                self.sl_order = None
                self.tp_order = None
                self.entry_price = None
                self.last_trade_bar = len(self.datas[0])

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE CLOSED: PnL={trade.pnlcomm:.2f}")

    def next(self):
        if self.order:
            return

        # Cooldown
        if (len(self.datas[0]) - self.last_trade_bar) < self.p.cooldown_bars:
            return

        price = float(self.data_close[0])
        rsi_val = float(self.rsi[0])
        bb_top = float(self.bb.top[0])
        bb_bot = float(self.bb.bot[0])
        vol = float(self.datas[0].volume[0])
        vol_avg = float(self.vol_sma[0]) if len(self.vol_sma) else 0

        # Volume filtresi
        has_volume = vol > (vol_avg * self.p.vol_mult) if vol_avg > 0 else True

        if not self.position:
            # LONG: RSI oversold + fiyat alt BB'ye dokundu + volume
            if rsi_val < self.p.rsi_oversold and price <= bb_bot and has_volume:
                self.order = self.buy()
                self.order.addinfo(tag="ENTRY")

            # SHORT: RSI overbought + fiyat üst BB'ye dokundu + volume
            elif (self.p.allow_short and
                  rsi_val > self.p.rsi_overbought and price >= bb_top and has_volume):
                self.order = self.sell()
                self.order.addinfo(tag="ENTRY_SHORT")

        else:
            # Ters sinyal çıkışı
            if self.position.size > 0 and rsi_val > self.p.rsi_overbought:
                self._cancel_bracket()
                self.order = self.close()
                self.order.addinfo(tag="EXIT")
            elif self.position.size < 0 and rsi_val < self.p.rsi_oversold:
                self._cancel_bracket()
                self.order = self.close()
                self.order.addinfo(tag="EXIT")
