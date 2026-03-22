# strategies/baseline_sma.py (v2)
# -*- coding: utf-8 -*-

import backtrader as bt


class BaselineSmaStrategy(bt.Strategy):
    """
    Sade benchmark stratejisi:
    - 5m kapanış üzerinde SMA(fast) & SMA(slow)
    - fast üstten keserse long aç
    - SL/TP broker stop/limit emirleri ile (intra-bar korumalı)
    - Ters kesişimde pozisyon kapat
    """

    params = dict(
        fast_period=10,
        slow_period=50,
        stop_loss=0.02,
        take_profit=0.05,
        allow_short=False,
    )

    def __init__(self):
        self.data_close = self.datas[0].close

        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data_close, period=int(self.p.fast_period)
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data_close, period=int(self.p.slow_period)
        )

        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        self.order = None
        self.sl_order = None
        self.tp_order = None
        self.entry_price = None

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} - {txt}")

    def _cancel_bracket(self):
        """Bekleyen SL/TP emirlerini iptal et."""
        for o in (self.sl_order, self.tp_order):
            if o is not None:
                try:
                    self.broker.cancel(o)
                except Exception:
                    pass
        self.sl_order = None
        self.tp_order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            tag = order.info.get("tag", "")

            # Long giriş tamamlandı → SL/TP emirlerini yerleştir
            if order.isbuy() and tag == "ENTRY":
                px = order.executed.price
                self.entry_price = px
                sl_price = px * (1.0 - float(self.p.stop_loss))
                tp_price = px * (1.0 + float(self.p.take_profit))

                self.sl_order = self.sell(
                    exectype=bt.Order.Stop, price=sl_price
                )
                self.sl_order.addinfo(tag="SL")
                self.tp_order = self.sell(
                    exectype=bt.Order.Limit, price=tp_price
                )
                self.tp_order.addinfo(tag="TP")

                self.log(
                    f"BUY EXECUTED @ {px:.2f}, "
                    f"SL={sl_price:.2f}, TP={tp_price:.2f}"
                )

            # SL veya TP tetiklendi
            elif order.issell() and tag in ("SL", "TP", "EXIT"):
                px = order.executed.price
                self.log(f"{tag} @ {px:.2f}")
                # Diğer bekleyen emri iptal et
                if tag == "SL" and self.tp_order:
                    try:
                        self.broker.cancel(self.tp_order)
                    except Exception:
                        pass
                    self.tp_order = None
                elif tag == "TP" and self.sl_order:
                    try:
                        self.broker.cancel(self.sl_order)
                    except Exception:
                        pass
                    self.sl_order = None

                self.entry_price = None

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"TRADE CLOSED, GROSS PNL={trade.pnl:.2f}, "
                f"NET PNL={trade.pnlcomm:.2f}"
            )

    def next(self):
        if self.order:
            return

        price = float(self.data_close[0])

        # Pozisyon yokken → giriş sinyali
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
                self.order.addinfo(tag="ENTRY")

        # Pozisyon varken → ters kesişimde çık (SL/TP zaten broker'da)
        else:
            if self.crossover < 0:
                self._cancel_bracket()
                self.order = self.close()
                self.order.addinfo(tag="EXIT")
                self.entry_price = None
