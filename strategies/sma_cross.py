# === strategies/sma_cross.py ===
import backtrader as bt

class SmaCrossStrategy(bt.Strategy):
    """
    YENİ: Multi-Timeframe (MTF) SMA Kesişim Stratejisi
    - datas[0] = Trade Timeframe (örn: 5m)
    - datas[1] = Trend Timeframe (örn: 1h)
    
    Strateji:
    1. 1h'lik trend yükselişteyse (hızlı > yavaş SMA)
    2. 5m'lik grafikte al sinyali (hızlı > yavaş SMA) ara.
    3. 5m'lik grafikte sat sinyali VEYA trendin dönmesi durumunda pozisyonu kapat.
    4. SL/TP yönetimi devam ediyor.
    """
    
    params = (
        ('trade_fast_sma', 10),
        ('trade_slow_sma', 50),
        ('trend_fast_sma', 10),
        ('trend_slow_sma', 50),
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
        ('printlog', True),
    )

    def __init__(self):
        print("--- MTF STRATEJİ __init__ BAŞLADI ---")
        
        self.data_trade = self.datas[0] # 5m'lik veri
        self.data_trend = self.datas[1] # 1h'lik veri

        self.order = None
        self.sl_order = None
        self.tp_order = None

        # --- 5m (Trade) İndikatörleri ---
        self.trade_fast = bt.indicators.SimpleMovingAverage(
            self.data_trade, period=self.p.trade_fast_sma
        )
        self.trade_slow = bt.indicators.SimpleMovingAverage(
            self.data_trade, period=self.p.trade_slow_sma
        )
        self.trade_crossover = bt.indicators.CrossOver(
            self.trade_fast, self.trade_slow
        )

        # --- 1h (Trend) İndikatörleri ---
        self.trend_fast = bt.indicators.SimpleMovingAverage(
            self.data_trend, period=self.p.trend_fast_sma
        )
        self.trend_slow = bt.indicators.SimpleMovingAverage(
            self.data_trend, period=self.p.trend_slow_sma
        )
        
        print("--- MTF STRATEJİ __init__ BİTTİ ---")

    def log(self, txt, dt=None, doprint=False):
        if self.p.printlog or doprint:
            dt = dt or self.data_trade.datetime.datetime(0) 
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                price = order.executed.price
                sl_price = price * (1.0 - self.p.stop_loss)
                tp_price = price * (1.0 + self.p.take_profit)
                
                self.sl_order = self.sell(data=self.data_trade, exectype=bt.Order.Stop, price=sl_price)
                self.tp_order = self.sell(data=self.data_trade, exectype=bt.Order.Limit, price=tp_price)
                self.log(f'ALIM GERÇEKLEŞTİ, Fiyat: {price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}')

            elif order.issell():
                self.log(f'SATIM GERÇEKLEŞTİ, Fiyat: {order.executed.price:.2f}')
                if self.sl_order:
                    self.broker.cancel(self.sl_order)
                    self.sl_order = None
                if self.tp_order:
                    self.broker.cancel(self.tp_order)
                    self.tp_order = None
            
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Emir İptal/Marj/Reddedildi')
            self.order = None
            self.sl_order = None
            self.tp_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERASYON KARI/ZARARI, Net {trade.pnlcomm:.2f}')

    def next(self):
        if self.order or self.sl_order or self.tp_order:
            return

        is_long_trend = self.trend_fast[0] > self.trend_slow[0]
        is_trade_entry = self.trade_crossover[0] > 0
        is_trade_exit = self.trade_crossover[0] < 0

        if not self.position:
            if is_long_trend and is_trade_entry:
                # === LOG DÜZELTMESİ BURADA ===
                self.log(f'ALIM SİNYALİ OLUŞTU (Trend UP, Trade Cross UP), Fiyat {self.data_trade.close[0]:.2f}')
                self.order = self.buy(data=self.data_trade)
        else:
            if not is_long_trend or is_trade_exit:
                # === VE BURADA ===
                self.log(f'SATIM SİNYALİ OLUŞTU (Trend döndü veya Trade Cross DOWN), Fiyat {self.data_trade.close[0]:.2f}')
                self.order = self.sell(data=self.data_trade)