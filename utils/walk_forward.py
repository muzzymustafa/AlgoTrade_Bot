# utils/walk_forward.py
"""
Rolling Walk-Forward Optimization framework.

Mantık:
- Veriyi zaman pencereleriyle kaydır (ör: 6 ay eğitim, 1 ay test)
- Her pencerede: parametre grid'i optimize et (eğitim setinde)
- En iyi parametrelerle test setinde backtest yap
- Tüm OOS (out-of-sample) sonuçları birleştir

Bu, klasik backtest'in en büyük zayıflığı olan overfitting'i ölçer.
Gerçek performans = OOS sonuçlarının ortalaması.
"""
import os
import pandas as pd
import numpy as np
import backtrader as bt
from dataclasses import dataclass

import config


@dataclass
class WFWindow:
    """Tek bir walk-forward penceresi."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict | None = None
    oos_metrics: dict | None = None


def generate_windows(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
) -> list[WFWindow]:
    """
    Rolling pencereler oluştur.

    Args:
        df: DatetimeIndex'li DataFrame
        train_bars: Eğitim penceresi bar sayısı
        test_bars: Test penceresi bar sayısı
        step_bars: Kaydırma adımı (None ise = test_bars)
    """
    if step_bars is None:
        step_bars = test_bars

    windows = []
    total = len(df)
    idx = df.index

    start = 0
    while start + train_bars + test_bars <= total:
        w = WFWindow(
            train_start=idx[start],
            train_end=idx[start + train_bars - 1],
            test_start=idx[start + train_bars],
            test_end=idx[min(start + train_bars + test_bars - 1, total - 1)],
        )
        windows.append(w)
        start += step_bars

    return windows


def _run_single_backtest(
    strategy_cls,
    df_trade: pd.DataFrame,
    start_date,
    end_date,
    bt_tf,
    bt_comp,
    params: dict,
) -> dict:
    """Tek bir cerebro çalıştır, metrikler döndür."""
    cerebro = bt.Cerebro(stdstats=False)

    data_feed = bt.feeds.PandasData(
        dataname=df_trade,
        fromdate=start_date,
        todate=end_date,
        timeframe=bt_tf,
        compression=bt_comp,
    )
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(config.START_CASH)
    cerebro.broker.setcommission(commission=config.COMMISSION_FEE)

    cerebro.addstrategy(strategy_cls, **params)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="ret")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="sharpe",
                        timeframe=bt_tf, compression=bt_comp, riskfreerate=0.0)

    results = cerebro.run()
    strat = results[0]

    ta = strat.analyzers.ta.get_analysis()
    ret = strat.analyzers.ret.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    total_trades = ta.get("total", {}).get("closed", 0) if ta else 0
    win_trades = ta.get("won", {}).get("total", 0) if ta else 0

    return {
        "end_value": float(cerebro.broker.getvalue()),
        "net_pl": float(cerebro.broker.getvalue()) - config.START_CASH,
        "rtot": ret.get("rtot", 0.0) if ret else 0.0,
        "max_dd": dd.get("max", {}).get("drawdown", 0.0) if dd else 0.0,
        "sharpe": sharpe.get("sharperatio", 0.0) if sharpe else 0.0,
        "total_trades": total_trades,
        "win_rate": (win_trades / total_trades) if total_trades > 0 else 0.0,
    }


def run_walk_forward(
    strategy_cls,
    df_trade: pd.DataFrame,
    param_grid: dict,
    train_bars: int = 50000,
    test_bars: int = 10000,
    step_bars: int | None = None,
    bt_tf=None,
    bt_comp: int = 5,
    optimize_metric: str = "sharpe",
) -> tuple[list[WFWindow], pd.DataFrame]:
    """
    Walk-forward optimization çalıştır.

    Args:
        strategy_cls: Backtrader strateji sınıfı
        df_trade: Tüm trade verisi
        param_grid: {'fast_period': [8,10,12], 'slow_period': [30,50], ...}
        train_bars: Eğitim penceresi
        test_bars: Test penceresi
        optimize_metric: Hangi metriğe göre optimize ('sharpe', 'net_pl', 'win_rate')

    Returns:
        (windows, oos_summary_df)
    """
    if bt_tf is None:
        bt_tf = bt.TimeFrame.Minutes

    windows = generate_windows(df_trade, train_bars, test_bars, step_bars)
    print(f"\n[Walk-Forward] {len(windows)} pencere oluşturuldu "
          f"(train={train_bars}, test={test_bars})")

    if not windows:
        print("[Walk-Forward] Yeterli veri yok!")
        return [], pd.DataFrame()

    # Parametre kombinasyonlarını üret
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]
    print(f"[Walk-Forward] {len(all_combos)} parametre kombinasyonu")

    oos_results = []

    for i, w in enumerate(windows):
        print(f"\n--- Pencere {i+1}/{len(windows)} ---")
        print(f"  Eğitim: {w.train_start} → {w.train_end}")
        print(f"  Test:   {w.test_start} → {w.test_end}")

        # 1. Eğitim: her parametre için backtest
        best_score = -np.inf
        best_params = all_combos[0]

        for combo in all_combos:
            try:
                metrics = _run_single_backtest(
                    strategy_cls, df_trade,
                    w.train_start, w.train_end,
                    bt_tf, bt_comp, combo,
                )
                score = metrics.get(optimize_metric, 0) or 0
                if score > best_score:
                    best_score = score
                    best_params = combo
            except Exception:
                continue

        w.best_params = best_params
        print(f"  En iyi params: {best_params} (score={best_score:.4f})")

        # 2. Test: en iyi parametrelerle OOS backtest
        try:
            oos = _run_single_backtest(
                strategy_cls, df_trade,
                w.test_start, w.test_end,
                bt_tf, bt_comp, best_params,
            )
            w.oos_metrics = oos
            oos_results.append({
                "window": i + 1,
                "test_start": str(w.test_start),
                "test_end": str(w.test_end),
                **best_params,
                **oos,
            })
            print(f"  OOS: net_pl={oos['net_pl']:.2f}, sharpe={oos['sharpe'] or 0:.3f}, "
                  f"trades={oos['total_trades']}")
        except Exception as e:
            print(f"  OOS hatası: {e}")

    # Özet
    df_oos = pd.DataFrame(oos_results)

    if not df_oos.empty:
        print(f"\n{'='*60}")
        print("  WALK-FORWARD ÖZET")
        print(f"{'='*60}")
        print(f"  Pencere sayısı: {len(df_oos)}")
        print(f"  Ort. OOS Net P/L: {df_oos['net_pl'].mean():.2f}")
        print(f"  Ort. OOS Sharpe:  {(df_oos['sharpe'].fillna(0)).mean():.3f}")
        print(f"  Ort. OOS Max DD:  {df_oos['max_dd'].mean():.2f}%")
        print(f"  Toplam OOS Trade: {df_oos['total_trades'].sum()}")
        print(f"  Kârlı pencere:   {(df_oos['net_pl'] > 0).sum()}/{len(df_oos)}")
        print(f"{'='*60}")

        # CSV kaydet
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "walk_forward_results.csv")
        df_oos.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSonuçlar: {csv_path}")

    return windows, df_oos
