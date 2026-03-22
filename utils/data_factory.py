from utils.data_provider import DataProvider


def get_provider(source: str, **kwargs) -> DataProvider:
    """Veri kaynağı adına göre uygun provider döndürür."""
    source = source.lower()

    if source == "binance":
        from utils.data_fetcher import BinanceProvider
        return BinanceProvider(**kwargs)
    elif source == "yfinance":
        from utils.yfinance_provider import YFinanceProvider
        return YFinanceProvider(**kwargs)
    else:
        raise ValueError(
            f"Bilinmeyen veri kaynağı: '{source}'. "
            f"Seçenekler: binance, yfinance"
        )
