# === config.py (v31) ===

# --- Borsa ve Veri Ayarları ---
SYMBOL = 'BTC/USDT'
TIMEFRAME_TREND = '1h'
TIMEFRAME_TRADE = '5m'
BARS_PER_REQUEST = 1000
TOTAL_BARS_TO_FETCH = 100000  # ~1 yıl

# --- Broker Ayarları ---
START_CASH = 10000.0
COMMISSION_FEE = 0.001
SIZER_PERCENTS = 95

# --- Strateji Parametreleri (Temel) ---
TREND_FAST_SMA = 10
TREND_SLOW_SMA = 50
TRADE_FAST_SMA = 10
TRADE_SLOW_SMA = 50

# --- Statik Risk (Yedek) ---
STOP_LOSS = 0.018
TAKE_PROFIT = 0.055

# -----------------------------------------------------------------
# === K-Means Ayarları ===
# -----------------------------------------------------------------
KMEANS_N_CLUSTERS = 3
KMEANS_ATR_PERIOD = 14
KMEANS_ROC_PERIOD = 20

# Otomatik iyi rejim seçimi (ROC'a göre)
KMEANS_AUTO_SELECT = True
GOOD_REGIME_ID = None

# -----------------------------------------------------------------
# === ARIMA Sinyal Filtresi Ayarları ===
# -----------------------------------------------------------------
ARIMA_ENABLED = False
ARIMA_ORDER = (1, 1, 0)
ARIMA_LOOKBACK = 50
ARIMA_FORECAST_STEPS = 3

# -----------------------------------------------------------------
# === ML & Dinamik Risk ===
# -----------------------------------------------------------------
USE_DYNAMIC_RISK = True
USE_META_LABELER = False
USE_BET_SIZING = False

ATR_PERIOD = 14
REGIME_SL_TP = {
    0: (1.0, 2.0),
    1: (2.0, 3.5),
    2: (1.5, 3.0),
}

# Temel filtreler (gevşek)
ADX_MIN = 15
MIN_CROSS_STRENGTH = 0.25
COOL_DOWN_BARS = 12
