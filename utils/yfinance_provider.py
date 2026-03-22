import yfinance as yf
import pandas as pd
from utils.data_provider import DataProvider


class YFinanceProvider(DataProvider):
    """
    yfinance veri kaynağı — BIST hisseleri ve diğer yfinance destekli piyasalar.

    Sembol formatı:
      - BIST: 'THYAO.IS', 'GARAN.IS', 'AKBNK.IS'
      - US: 'AAPL', 'MSFT'
      - Crypto (yfinance): 'BTC-USD'

    Kısıtlamalar:
      - 1m veri: max 7 gün geriye
      - 5m/15m/30m: max 60 gün
      - 1h: max ~730 gün
      - 1d: sınırsız (geçmiş veriye kadar)
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }

    # Her timeframe için yfinance'in izin verdiği max gün sayısı
    MAX_DAYS = {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "1d": 10000,
        "1w": 10000,
        "1M": 10000,
    }

    # Her timeframe'deki bar/gün oranı (yaklaşık)
    BARS_PER_DAY = {
        "1m": 390,   # hisse senedi saatleri (~6.5 saat)
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "1h": 7,     # hisse saatleri (~7 saat)
        "1d": 1,
        "1w": 0.2,
        "1M": 1 / 30,
    }

    def name(self) -> str:
        return "yfinance"

    def supported_timeframes(self) -> list:
        return list(self.TIMEFRAME_MAP.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        total_bars: int,
        bars_per_request: int = 1000,
    ) -> pd.DataFrame:
        yf_interval = self.TIMEFRAME_MAP.get(timeframe)
        if yf_interval is None:
            raise ValueError(
                f"Desteklenmeyen timeframe: {timeframe}. "
                f"Seçenekler: {list(self.TIMEFRAME_MAP.keys())}"
            )

        # Kaç gün geriye gitmemiz gerektiğini hesapla
        bpd = self.BARS_PER_DAY.get(timeframe, 1)
        days_needed = int(total_bars / max(bpd, 0.01)) + 5  # biraz marj
        max_days = self.MAX_DAYS.get(timeframe, 10000)
        days_to_fetch = min(days_needed, max_days)

        print(f"[yfinance] {symbol} için {timeframe} verisi çekiliyor "
              f"(~{days_to_fetch} gün)...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days_to_fetch}d", interval=yf_interval)

        if df is None or df.empty:
            print(f"[yfinance] {symbol} için veri bulunamadı!")
            return pd.DataFrame()

        # Sütun adlarını normalize et
        df.columns = [c.lower() for c in df.columns]

        # Sadece OHLCV sütunlarını tut
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols]
        df.index.name = "datetime"

        # Timezone bilgisini kaldır (backtrader uyumluluğu)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        final_df = df.tail(total_bars)
        print(f"[yfinance] {len(final_df)} bar alındı "
              f"({final_df.index.min()} - {final_df.index.max()})")
        return final_df
