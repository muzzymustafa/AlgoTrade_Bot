# run_optimization.py — BIST hisseleri üzerinde kapsamlı optimizasyon
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import backtrader as bt
import pandas as pd
import numpy as np
from itertools import product
from utils.yfinance_provider import YFinanceProvider
from strategies.baseline_sma import BaselineSmaStrategy
from strategies.rsi_bollinger import RsiBollingerStrategy

SYMBOLS = ['THYAO.IS', 'GARAN.IS', 'AKBNK.IS']
START_CASH = 100000
COMMISSION = 0.002  # BIST komisyon

provider = YFinanceProvider()


def run_single(strategy_cls, df, params, start_cash=START_CASH):
    """Tek parametre seti ile backtest çalıştır."""
    cerebro = bt.Cerebro(stdstats=False)
    data_feed = bt.feeds.PandasData(
        dataname=df, fromdate=df.index.min(), todate=df.index.max(),
        timeframe=bt.TimeFrame.Days, compression=1,
    )
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Bakiyenin %30'u ile pozisyon aç (BIST için makul)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=30)

    cerebro.addstrategy(strategy_cls, **params)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='ret')

    try:
        results = cerebro.run()
    except Exception:
        return None

    strat = results[0]
    ta = strat.analyzers.ta.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    ret = strat.analyzers.ret.get_analysis()

    total = ta.get('total', {}).get('closed', 0) if ta else 0
    won = ta.get('won', {}).get('total', 0) if ta else 0

    return {
        **params,
        'total_trades': total,
        'win_rate': won / total if total > 0 else 0,
        'net_pl': cerebro.broker.getvalue() - start_cash,
        'return_pct': (ret.get('rtot', 0) or 0) * 100,
        'max_dd': dd.get('max', {}).get('drawdown', 0) if dd else 0,
        'sharpe': sharpe.get('sharperatio', 0) or 0,
        'end_value': cerebro.broker.getvalue(),
    }


def optimize_sma(df, symbol):
    """SMA Crossover grid optimizasyonu."""
    print(f"\n{'='*60}")
    print(f"  SMA OPTİMİZASYON: {symbol}")
    print(f"{'='*60}")

    grid = list(product(
        [5, 8, 10, 15, 20],        # fast_period
        [30, 40, 50, 60, 80],       # slow_period
        [0.02, 0.03, 0.04, 0.05],   # stop_loss
        [0.04, 0.06, 0.08, 0.10],   # take_profit
    ))
    # fast < slow filtresi
    grid = [(f, s, sl, tp) for f, s, sl, tp in grid if f < s]

    print(f"  {len(grid)} kombinasyon test edilecek...")
    rows = []
    for i, (fast, slow, sl, tp) in enumerate(grid):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}...")
        params = {'fast_period': fast, 'slow_period': slow,
                  'stop_loss': sl, 'take_profit': tp}
        r = run_single(BaselineSmaStrategy, df, params)
        if r and r['total_trades'] >= 3:
            rows.append(r)

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        print("  Sonuç yok!")
        return df_res

    df_res = df_res.sort_values('net_pl', ascending=False)
    print(f"\n  TOP 5 (Net P/L):")
    for _, row in df_res.head(5).iterrows():
        print(f"    SMA({int(row['fast_period'])},{int(row['slow_period'])}) "
              f"SL={row['stop_loss']:.0%} TP={row['take_profit']:.0%} | "
              f"P/L={row['net_pl']:+,.0f} TL | WR={row['win_rate']:.0%} | "
              f"DD={row['max_dd']:.1f}% | Sharpe={row['sharpe']:.2f} | "
              f"Trades={int(row['total_trades'])}")
    return df_res


def optimize_rsi_bb(df, symbol):
    """RSI + Bollinger grid optimizasyonu."""
    print(f"\n{'='*60}")
    print(f"  RSI+BB OPTİMİZASYON: {symbol}")
    print(f"{'='*60}")

    grid = list(product(
        [10, 14, 20],           # rsi_period
        [25, 30, 35],           # rsi_oversold
        [65, 70, 75],           # rsi_overbought
        [15, 20, 25],           # bb_period
        [1.5, 2.0, 2.5],       # bb_devfactor
    ))

    print(f"  {len(grid)} kombinasyon test edilecek...")
    rows = []
    for i, (rsi_p, rsi_os, rsi_ob, bb_p, bb_dev) in enumerate(grid):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}...")
        params = {
            'rsi_period': rsi_p, 'rsi_oversold': rsi_os, 'rsi_overbought': rsi_ob,
            'bb_period': bb_p, 'bb_devfactor': bb_dev,
            'stop_loss': 0.03, 'take_profit': 0.06,
        }
        r = run_single(RsiBollingerStrategy, df, params)
        if r and r['total_trades'] >= 3:
            rows.append(r)

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        print("  Sonuç yok!")
        return df_res

    df_res = df_res.sort_values('net_pl', ascending=False)
    print(f"\n  TOP 5 (Net P/L):")
    for _, row in df_res.head(5).iterrows():
        print(f"    RSI({int(row['rsi_period'])}) OS={int(row['rsi_oversold'])} "
              f"OB={int(row['rsi_overbought'])} BB({int(row['bb_period'])},{row['bb_devfactor']:.1f}) | "
              f"P/L={row['net_pl']:+,.0f} TL | WR={row['win_rate']:.0%} | "
              f"DD={row['max_dd']:.1f}% | Sharpe={row['sharpe']:.2f} | "
              f"Trades={int(row['total_trades'])}")
    return df_res


# === ÇALIŞTIR ===
os.makedirs('results', exist_ok=True)
all_results = []

for sym in SYMBOLS:
    print(f"\n{'#'*60}")
    print(f"  {sym} — Veri çekiliyor...")
    print(f"{'#'*60}")

    df = provider.fetch_ohlcv(sym, '1d', 1000)
    if df.empty:
        print(f"  {sym} verisi alınamadı!")
        continue

    # SMA optimizasyon
    sma_res = optimize_sma(df, sym)
    if not sma_res.empty:
        sma_res['symbol'] = sym
        sma_res['strategy'] = 'SMA'
        sma_res.to_csv(f'results/opt_{sym.replace(".","_")}_sma.csv', index=False)

    # RSI+BB optimizasyon
    rsi_res = optimize_rsi_bb(df, sym)
    if not rsi_res.empty:
        rsi_res['symbol'] = sym
        rsi_res['strategy'] = 'RSI+BB'
        rsi_res.to_csv(f'results/opt_{sym.replace(".","_")}_rsi.csv', index=False)

    # En iyi sonuçları topla
    if not sma_res.empty:
        best_sma = sma_res.iloc[0].to_dict()
        best_sma['symbol'] = sym
        best_sma['strategy'] = 'SMA'
        all_results.append(best_sma)
    if not rsi_res.empty:
        best_rsi = rsi_res.iloc[0].to_dict()
        best_rsi['symbol'] = sym
        best_rsi['strategy'] = 'RSI+BB'
        all_results.append(best_rsi)

# === GENEL KARŞILAŞTIRMA ===
print(f"\n{'='*70}")
print(f"  GENEL KARŞILAŞTIRMA — EN İYİ SONUÇLAR")
print(f"{'='*70}")

df_all = pd.DataFrame(all_results)
if not df_all.empty:
    for _, row in df_all.iterrows():
        print(f"  {row['symbol']:12s} | {row['strategy']:6s} | "
              f"P/L={row['net_pl']:+10,.0f} TL | "
              f"Return={row['return_pct']:+6.1f}% | "
              f"WR={row['win_rate']:.0%} | "
              f"DD={row['max_dd']:.1f}% | "
              f"Sharpe={row['sharpe']:+.2f} | "
              f"Trades={int(row['total_trades'])}")

    df_all.to_csv('results/best_results_comparison.csv', index=False)
    print(f"\nDetaylı sonuçlar: results/ dizini")
print(f"{'='*70}")
