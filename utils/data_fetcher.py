# === utils/data_fetcher.py ===
import ccxt
import pandas as pd
import time

from utils.data_provider import DataProvider


class BinanceProvider(DataProvider):
    """Binance CCXT veri kaynağı — kripto çiftleri için."""

    def __init__(self, rate_limit=1200):
        self.exchange = ccxt.binance({
            "rateLimit": rate_limit,
            "enableRateLimit": True,
        })

    def name(self) -> str:
        return "binance"

    def supported_timeframes(self) -> list:
        return [
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "1w",
        ]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        total_bars: int,
        bars_per_request: int = 1000,
    ) -> pd.DataFrame:
        print(f"[Binance] {symbol} için {timeframe} verisi çekiliyor...")
        print(f"  Toplam {total_bars} bar, {bars_per_request} bar/istek")

        timeframe_duration_in_ms = self.exchange.parse_timeframe(timeframe) * 1000
        all_ohlcv = []

        try:
            since = self.exchange.milliseconds() - total_bars * timeframe_duration_in_ms

            while len(all_ohlcv) < total_bars:
                print(f"  Çekilen: {len(all_ohlcv)} / {total_bars}")

                limit = min(total_bars - len(all_ohlcv), bars_per_request)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + timeframe_duration_in_ms
                time.sleep(self.exchange.rateLimit / 1000)

            print(f"[Binance] Toplam {len(all_ohlcv)} bar alındı.")

            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df.drop_duplicates(subset="timestamp", inplace=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            df.drop("timestamp", axis=1, inplace=True)

            final_df = df.tail(total_bars)
            print(f"[Binance] DataFrame: {len(final_df)} satır")
            return final_df

        except Exception as e:
            print(f"[Binance] Veri çekme hatası: {e}")
            return pd.DataFrame()


# Backward compatibility wrapper
def fetch_binance_data(symbol, timeframe, total_bars, bars_per_request):
    """Eski API uyumluluğu için wrapper."""
    provider = BinanceProvider()
    result = provider.fetch_ohlcv(symbol, timeframe, total_bars, bars_per_request)
    if result.empty:
        return None
    return result
