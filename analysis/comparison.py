# analysis/comparison.py
"""Multi-symbol backtest sonuçlarını karşılaştır."""
import os
import pandas as pd


def print_comparison_table(results: dict, save_csv: bool = True):
    """
    Birden fazla sembolün backtest sonuçlarını tablo halinde göster.

    Args:
        results: {symbol: metrics_dict} formatında sonuçlar
        save_csv: True ise CSV'ye kaydet
    """
    rows = []
    for symbol, metrics in results.items():
        if metrics is None:
            rows.append({"Symbol": symbol, "Durum": "HATA/VERİ YOK"})
            continue
        rows.append({
            "Symbol": symbol,
            "Kaynak": metrics.get("source", "?"),
            "Net P/L": f"{metrics.get('net_pl', 0):.2f}",
            "Return %": f"{metrics.get('rtot', 0) * 100:.2f}%",
            "Sharpe": f"{metrics.get('sharpe', 0) or 0:.3f}",
            "Max DD %": f"{metrics.get('max_dd', 0):.2f}%",
            "SQN": f"{metrics.get('sqn', 0) or 0:.2f}",
            "Trades": metrics.get("total_trades", 0),
            "Win Rate": f"{metrics.get('win_rate', 0) * 100:.1f}%",
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("  MULTI-SYMBOL BACKTEST KARŞILAŞTIRMASI")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    if save_csv:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "multi_symbol_comparison.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSonuçlar kaydedildi: {csv_path}")

    return df
