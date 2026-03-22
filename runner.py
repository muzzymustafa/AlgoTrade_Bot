# runner.py — Multi-symbol backtest orchestrator
"""
Her sembol için bağımsız cerebro çalıştırır ve sonuçları toplar.

Kullanım:
    python main.py --multi --mode baseline
    python main.py --multi --mode ml
"""
import config
from analysis.comparison import print_comparison_table


def _snapshot_config():
    """Config'in mevcut durumunu kaydet."""
    return {
        "SYMBOL": config.SYMBOL,
        "DATA_SOURCE": config.DATA_SOURCE,
        "TIMEFRAME_TRADE": config.TIMEFRAME_TRADE,
        "TIMEFRAME_TREND": config.TIMEFRAME_TREND,
        "TOTAL_BARS_TO_FETCH": config.TOTAL_BARS_TO_FETCH,
        "FILTER_TRADING_HOURS": getattr(config, "FILTER_TRADING_HOURS", True),
    }


def _restore_config(snapshot: dict):
    """Config'i önceki durumuna geri yükle."""
    for key, val in snapshot.items():
        setattr(config, key, val)


def _apply_symbol_config(sym_cfg: dict):
    """Tek bir sembolün ayarlarını config'e uygula."""
    config.SYMBOL = sym_cfg["symbol"]
    config.DATA_SOURCE = sym_cfg.get("source", "binance")
    config.TIMEFRAME_TRADE = sym_cfg.get("timeframe_trade", "5m")
    config.TIMEFRAME_TREND = sym_cfg.get("timeframe_trend", "1h")
    config.TOTAL_BARS_TO_FETCH = sym_cfg.get("total_bars", config.TOTAL_BARS_TO_FETCH)
    config.FILTER_TRADING_HOURS = sym_cfg.get("filter_trading_hours", True)


def run_multi_symbol(mode: str, optimize: bool = False, full_backtest: bool = False):
    """
    config.SYMBOLS listesindeki her sembol için backtest çalıştır.

    Args:
        mode: 'baseline' veya 'ml'
        optimize: optimizasyon modu
        full_backtest: ML modunda tüm veri üzerinde çalıştır
    """
    # Lazy import — circular import'u önler
    from main import run_baseline, run_ml

    symbols = getattr(config, "SYMBOLS", [])
    if not symbols:
        print("config.SYMBOLS listesi boş! En az bir sembol tanımlayın.")
        return

    print(f"\n{'='*60}")
    print(f"  MULTI-SYMBOL MODU: {len(symbols)} sembol, mod={mode}")
    print(f"{'='*60}")

    results = {}
    original = _snapshot_config()

    for i, sym_cfg in enumerate(symbols, 1):
        symbol = sym_cfg["symbol"]
        source = sym_cfg.get("source", "binance")

        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(symbols)}] {symbol} ({source})")
        print(f"{'─'*60}")

        try:
            _apply_symbol_config(sym_cfg)

            if mode == "baseline":
                result = run_baseline(optimize=optimize)
            else:
                result = run_ml(optimize=optimize, full_backtest=full_backtest)

            results[symbol] = result

        except Exception as e:
            print(f"\n!!! {symbol} backtest hatası: {e}")
            results[symbol] = None

        finally:
            _restore_config(original)

    # Karşılaştırma tablosu
    print_comparison_table(results)
    return results
