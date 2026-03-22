from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """Tüm veri kaynaklarının uygulaması gereken arayüz."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        total_bars: int,
        bars_per_request: int = 1000,
    ) -> pd.DataFrame:
        """
        OHLCV verisi döndürür.

        Returns:
            DataFrame: DatetimeIndex, columns=[open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def supported_timeframes(self) -> list:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
