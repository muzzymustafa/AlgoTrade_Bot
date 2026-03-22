# tests/test_multi_symbol.py
"""Multi-symbol config ve runner testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_symbols_config_structure():
    """Her sembol config'inde gerekli alanlar olmalı."""
    import config
    required_keys = {"symbol", "source", "timeframe_trade", "timeframe_trend"}

    for i, sym in enumerate(config.SYMBOLS):
        missing = required_keys - set(sym.keys())
        assert not missing, f"SYMBOLS[{i}] ({sym.get('symbol','?')}) eksik alanlar: {missing}"


def test_symbols_valid_sources():
    """Veri kaynağı binance veya yfinance olmalı."""
    import config
    valid_sources = {"binance", "yfinance"}

    for sym in config.SYMBOLS:
        assert sym.get("source") in valid_sources, \
            f"{sym['symbol']}: geçersiz kaynak '{sym.get('source')}'"


def test_bist_symbols_use_yfinance():
    """'.IS' uzantılı semboller yfinance kullanmalı."""
    import config

    for sym in config.SYMBOLS:
        if sym["symbol"].endswith(".IS"):
            assert sym["source"] == "yfinance", \
                f"{sym['symbol']} BIST hissesi ama kaynak '{sym['source']}'"


def test_bist_no_trading_hours_filter():
    """BIST sembolleri için saat filtresi kapalı olmalı."""
    import config

    for sym in config.SYMBOLS:
        if sym["symbol"].endswith(".IS"):
            assert sym.get("filter_trading_hours", True) is False, \
                f"{sym['symbol']} BIST hissesi ama saat filtresi açık"


def test_snapshot_restore():
    """Config snapshot/restore düzgün çalışmalı."""
    import config
    from runner import _snapshot_config, _restore_config, _apply_symbol_config

    original = _snapshot_config()

    # Farklı sembol uygula
    _apply_symbol_config({
        "symbol": "TEST/USD",
        "source": "yfinance",
        "timeframe_trade": "1d",
        "timeframe_trend": "1w",
    })

    assert config.SYMBOL == "TEST/USD"
    assert config.DATA_SOURCE == "yfinance"

    # Geri yükle
    _restore_config(original)

    assert config.SYMBOL == original["SYMBOL"]
    assert config.DATA_SOURCE == original["DATA_SOURCE"]


def test_comparison_table():
    """Karşılaştırma tablosu boş sonuçları handle etmeli."""
    from analysis.comparison import print_comparison_table

    results = {
        "BTC/USDT": {
            "source": "binance", "net_pl": 500, "rtot": 0.05,
            "sharpe": 1.2, "max_dd": 5.0, "sqn": 2.1,
            "total_trades": 50, "win_rate": 0.6,
        },
        "THYAO.IS": None,  # hata durumu
    }

    df = print_comparison_table(results, save_csv=False)
    assert len(df) == 2
    assert "HATA" in df.iloc[1].to_string()
