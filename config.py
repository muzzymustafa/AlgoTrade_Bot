# === config.py (v34) ===
# -----------------------------------------------------------------
# --- Veri Kaynağı ve Borsa Ayarları ---
# -----------------------------------------------------------------
DATA_SOURCE = 'binance'   # 'binance' | 'yfinance'

SYMBOL = 'BTC/USDT'
# BIST örnekleri: 'THYAO.IS', 'GARAN.IS', 'AKBNK.IS'
# Crypto: 'BTC/USDT', 'ETH/USDT'

TIMEFRAME_TREND = '1h'   # Üst zaman dilimi (HTF) - trend filtresi için
TIMEFRAME_TRADE = '5m'   # İşlem zaman dilimi (LTF)

BARS_PER_REQUEST = 1000
TOTAL_BARS_TO_FETCH = 100000  # ~1 yıl (5m için)

# HTF filtresi (ör: 5m -> 1h = 5 * 12 = 60 dakika)
HTF_MULTIPLIER = 12
USE_HTF_FILTER = True

# Saat filtresi — crypto için True, hisse (BIST) için False
FILTER_TRADING_HOURS = True

# -----------------------------------------------------------------
# --- Broker Ayarları ---
# -----------------------------------------------------------------
START_CASH = 10000.0
COMMISSION_FEE = 0.001      # tek yön komisyon (örn. 10 bps = 0.001)
SLIPPAGE_BPS = 2            # tek yön slippage (2 bps = 0.0002 fiyat çarpanı)
SIZER_PERCENTS = 20         # PercentSizer için yüzde (güvenli fallback)
USE_ATR_POSITION_SIZING = True
RISK_PER_TRADE = 0.005      # bakiyenin %0.5’i risk/işlem (ATR tabanlı sizing)

# -----------------------------------------------------------------
# --- Strateji Parametreleri (ML modundaki SMA Cross) ---
# -----------------------------------------------------------------
TREND_FAST_SMA = 10
TREND_SLOW_SMA = 50

TRADE_FAST_SMA = 10
TRADE_SLOW_SMA = 50

HOLD_BARS_MIN = 3           # minimum elde tutma süresi (mikro gürültüden kaç)

# --- Statik Risk (Yedek / fallback) ---
STOP_LOSS = 0.018
TAKE_PROFIT = 0.055

# -----------------------------------------------------------------
# --- Baseline SMA Benchmark Parametreleri ---
# -----------------------------------------------------------------
# Optimizasyon sonrası "resmi" benchmark olarak kullanacağın değerleri
# buraya sabitleyeceğiz. Şimdilik mevcut default’ları kullanıyoruz.
BASELINE_TRADE_FAST_SMA = 10
BASELINE_TRADE_SLOW_SMA = 50
BASELINE_STOP_LOSS = 0.018
BASELINE_TAKE_PROFIT = 0.055

# -----------------------------------------------------------------
# --- Baseline Optimizasyon Grid’i (SMA + SL/TP) ---
# -----------------------------------------------------------------
# python main.py --mode baseline --optimize
# çalıştırıldığında kullanılacak grid.
OPT_TRADE_FAST_SMA = [5, 8, 10, 12, 15]
OPT_TRADE_SLOW_SMA = [30, 40, 50, 60, 80]
OPT_STOP_LOSS      = [0.010, 0.015, 0.020, 0.025]
OPT_TAKE_PROFIT    = [0.020, 0.030, 0.040, 0.050]

# -----------------------------------------------------------------
# === K-Means Ayarları (ML modu için) ===
# -----------------------------------------------------------------
KMEANS_N_CLUSTERS = 3
KMEANS_ATR_PERIOD = 14
KMEANS_ROC_PERIOD = 20

# Otomatik iyi rejim seçimi (ROC'a göre) + .env ile override
KMEANS_AUTO_SELECT = True
GOOD_REGIME_ID = None  # .env'de GOOD_REGIME_ID varsa kod bunu almalı

# Rejim filtresi için z-skor eşikleri
ROC_HIGH_Z = 0.8                   # momentumun belirgin olduğu rejimler
ATR_Z_MIN, ATR_Z_MAX = -0.3, 1.2   # çok ölü / çok panik pazarları ele

# -----------------------------------------------------------------
# === ARIMA Sinyal Filtresi Ayarları (opsiyonel, ML modu) ===
# -----------------------------------------------------------------
ARIMA_ENABLED = False
ARIMA_ORDER = (1, 1, 0)
ARIMA_LOOKBACK = 50
ARIMA_FORECAST_STEPS = 3

# -----------------------------------------------------------------
# === ML & Dinamik Risk (ML modu) ===
# -----------------------------------------------------------------
USE_DYNAMIC_RISK = True   # OTR / rejime göre SL/TP vs.
USE_META_LABELER = True
META_THRESHOLD = 0.60     # meta-olumlu olasılık eşiği (0-1)
USE_BET_SIZING = False

ATR_PERIOD = 14
REGIME_SL_TP = {
    0: (1.0, 2.0),
    1: (2.0, 3.5),
    2: (1.5, 3.0),
}

# -----------------------------------------------------------------
# --- Temel Filtreler (ML modundaki feature-based sinyal sistemi) ---
# -----------------------------------------------------------------
ADX_MIN = 20                 # trend gücü eşiği
MIN_CROSS_STRENGTH = 0.25
COOL_DOWN_BARS = 12          # yeniden giriş için bekleme

# -----------------------------------------------------------------
# --- Multi-Symbol Modu ---
# -----------------------------------------------------------------
MULTI_SYMBOL_MODE = False

SYMBOLS = [
    {
        'symbol': 'BTC/USDT',
        'source': 'binance',
        'timeframe_trade': '5m',
        'timeframe_trend': '1h',
        'total_bars': 100000,
        'filter_trading_hours': True,
    },
    {
        'symbol': 'ETH/USDT',
        'source': 'binance',
        'timeframe_trade': '5m',
        'timeframe_trend': '1h',
        'total_bars': 100000,
        'filter_trading_hours': True,
    },
    {
        'symbol': 'THYAO.IS',
        'source': 'yfinance',
        'timeframe_trade': '1d',
        'timeframe_trend': '1w',
        'total_bars': 1000,
        'filter_trading_hours': False,
    },
    {
        'symbol': 'GARAN.IS',
        'source': 'yfinance',
        'timeframe_trade': '1d',
        'timeframe_trend': '1w',
        'total_bars': 1000,
        'filter_trading_hours': False,
    },
]
