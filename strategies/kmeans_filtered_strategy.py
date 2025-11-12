# === strategies/ml_strategy.py ===
# (Eski adı: kmeans_filtered_strategy.py)
import backtrader as bt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Statsmodels'un üretebileceği uyarıları bastır
warnings.filterwarnings("ignore")

class MlStrategy(bt.Strategy):
    """
    K-Means Rejim Filtreli ve ARIMA Sinyal Filtreli MTF Stratejisi
    - datas[0] = Trade Timeframe (5m)
    - datas[1] = Trend Timeframe (1h) + K-Means Rejim verisi
    """
    
    params = (
        ('trade_fast_sma', 10),
        ('trade_slow_sma', 50),
        ('trend_fast_sma', 10),
        ('trend_slow_sma', 50),
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
        ('good_regime_id', 0),
        # YENİ: ARIMA Parametreleri
        ('arima_enabled', True),
        ('arima_order', (1, 1, 0)),
        ('arima_lookback', 50),
        ('arima_forecast_steps', 3),
        ('printlog', True),
    )

    def __init__(self):
        print("--- ML Strateji __init__ BAŞLADI (K-Means + ARIMA) ---")
        
        self.data_trade = self.datas[0] # 5m
        self.data_trend = self.datas[1] # 1h
        
        self.order = None
        self.sl_order = None
        self.tp_order = None

        # --- 5m (Trade) İndikatörleri ---
        self.trade_fast = bt.indicators.SimpleMovingAverage(self.data_trade, period=self.p.trade_fast_sma)
        self.trade_slow = bt.indicators.SimpleMovingAverage(self.data_trade, period=self.p.trade_slow_sma)
        self.trade_crossover = bt.indicators.CrossOver(self.trade_fast, self.trade_slow)

        # --- 1h (Trend) İndikatörleri ---
        self.trend_fast = bt.indicators.SimpleMovingAverage(self.data_trend, period=self.p.trend_fast_sma)
        self.trend_slow = bt.indicators.SimpleMovingAverage(self.data_trend, period=self.p.trend_slow_sma)
        self.current_regime = self.data_trend.lines.regime # K-Means hattı
        
        print(f"--- ML Strateji __init__ BİTTİ. İyi Rejim: {self.p.good_regime_id} ---")

    def log(self, txt, dt=None, doprint=False):
        if self.p.printlog or doprint:
            dt = dt or self.data_trade.datetime.datetime(0) 
            print(f'{dt.isoformat()}, {txt}')

    # --- YENİ: ARIMA TAHMİN FONKSİYONU ---
    def get_arima_forecast(self):
        """
        'self.data_trade' verisinin son 'arima_lookback' barını kullanarak
        'arima_forecast_steps' adım sonrasını tahmin eder.
        Tahmin yükseliş yönlüyse True döndürür.
        """
        try:
            # 1. Modeli eğitmek için veriyi al
            # .get(size=N) metodu, son N barı bir liste olarak döndürür
            data = self.data_trade.close.get(size=self.p.arima_lookback)
            
            # Veri yetersizse (backtest'in başındaysak)
            if len(data) < self.p.arima_lookback:
                return False # Güvenli tarafta kal, sinyali engelle

            # 2. Modeli oluştur ve eğit
            model = ARIMA(data, order=self.p.arima_order)
            # 'full_output=False' ve 'disp=0' hata ayıklama çıktılarını gizler
            model_fit = model.fit() 

            # 3. Tahmin yap
            forecast = model_fit.forecast(steps=self.p.arima_forecast_steps)
            
            # 4. Karar ver
            # Tahmin edilen son fiyat (örn: 3 adım sonraki), şu anki fiyattan yüksek mi?
            current_price = data[-1]
            forecasted_price = forecast[-1] # Tahmin dizisinin son elemanı
            
            if forecasted_price > current_price:
                return True # Tahmin yükseliş yönlü
            else:
                return False # Tahmin yatay veya düşüş yönlü

        except Exception as e:
            # Modelin eğitilmesi (convergence hatası vb.) başarısız olursa
            self.log(f'ARIMA modeli eğitimi başarısız oldu: {e}')
            return False # Güvenli tarafta kal, sinyali engelle

    # --- notify_order ve notify_trade metodları (DEĞİŞİKLİK YOK) ---
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
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
                if self.sl_order: self.broker.cancel(self.sl_order); self.sl_order = None
                if self.tp_order: self.broker.cancel(self.tp_order); self.tp_order = None
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Emir İptal/Marj/Reddedildi')
            self.order = None; self.sl_order = None; self.tp_order = None
    def notify_trade(self, trade):
        if not trade.isclosed: return
        self.log(f'OPERASYON KARI/ZARARI, Net {trade.pnlcomm:.2f}')
    # --- Bitiş (DEĞİŞİKLİK YOK) ---


    def next(self):
        if self.order or self.sl_order or self.tp_order:
            return # Bekleyen emir varsa çık

        # 1. Trend Filtresi (1h)
        is_long_trend = self.trend_fast[0] > self.trend_slow[0]
        # 2. Rejim Filtresi (1h)
        current_regime_label = self.current_regime[0]
        is_good_regime = (current_regime_label == self.p.good_regime_id)
        # 3. Trade Sinyali (5m)
        is_trade_entry = self.trade_crossover[0] > 0
        is_trade_exit = self.trade_crossover[0] < 0

        if not self.position:
            # ALIM KOŞULU (3 AŞAMALI FİLTRE)
            if is_long_trend and is_good_regime and is_trade_entry:
                
                # --- YENİ: 4. ARIMA FİLTRESİ ---
                is_arima_confirmed = True # Varsayılan (eğer ARIMA kapalıysa)
                if self.p.arima_enabled:
                    self.log('Sinyal alındı, ARIMA tahmini bekleniyor...')
                    is_arima_confirmed = self.get_arima_forecast()
                
                if is_arima_confirmed:
                    self.log(f'ALIM SİNYALİ OLUŞTU (Trend UP, Rejim {current_regime_label} OK, ARIMA OK)')
                    self.order = self.buy(data=self.data_trade)
                else:
                    self.log(f'Sinyal geldi ama ARIMA tarafından (düşüş/yatay) engellendi.')
                
            elif is_long_trend and is_trade_entry and not is_good_regime:
                self.log(f'Sinyal geldi ama Rejim {current_regime_label} tarafından engellendi.')

        # SATIM KOŞULU (Değişiklik yok)
        else:
            if not is_long_trend or is_trade_exit:
                self.log(f'STRATEJİ SATIM SİNYALİ OLUŞTU (Trend döndü veya Trade Cross DOWN)')
                self.order = self.sell(data=self.data_trade)