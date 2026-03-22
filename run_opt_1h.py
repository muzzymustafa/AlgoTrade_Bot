# run_opt_1h.py — Saatlik BIST optimizasyonu (log'suz, hızlı)
import sys, os, io, contextlib
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
COMMISSION = 0.002
provider = YFinanceProvider()


def run_silent(strategy_cls, df, params):
    """Backtest — tüm stdout susturulmuş."""
    cerebro = bt.Cerebro(stdstats=False)
    data_feed = bt.feeds.PandasData(
        dataname=df, fromdate=df.index.min(), todate=df.index.max(),
        timeframe=bt.TimeFrame.Minutes, compression=60,
    )
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=30)
    cerebro.addstrategy(strategy_cls, **params)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe',
                        timeframe=bt.TimeFrame.Minutes, compression=60, riskfreerate=0.0)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = cerebro.run()
    except Exception:
        return None

    strat = results[0]
    ta = strat.analyzers.ta.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    total = ta.get('total', {}).get('closed', 0) if ta else 0
    won = ta.get('won', {}).get('total', 0) if ta else 0

    return {
        **params,
        'total_trades': total,
        'win_rate': won / total if total > 0 else 0,
        'net_pl': cerebro.broker.getvalue() - START_CASH,
        'return_pct': (cerebro.broker.getvalue() / START_CASH - 1) * 100,
        'max_dd': dd.get('max', {}).get('drawdown', 0) if dd else 0,
        'sharpe': sharpe.get('sharperatio', 0) or 0,
    }


os.makedirs('results', exist_ok=True)
all_best = []

for sym in SYMBOLS:
    print(f'\n{"#"*60}')
    print(f'  {sym}')
    print(f'{"#"*60}')
    df = provider.fetch_ohlcv(sym, '1h', 5000)
    if df.empty:
        continue
    print(f'  {len(df)} bar ({df.index.min().date()} - {df.index.max().date()})')

    # === SMA ===
    sma_grid = [(f, s, sl, tp) for f, s, sl, tp in product(
        [3, 5, 8, 10, 15],
        [15, 20, 30, 40, 50],
        [0.01, 0.015, 0.02, 0.03],
        [0.02, 0.03, 0.04, 0.06],
    ) if f < s]

    print(f'  SMA: {len(sma_grid)} kombinasyon...', end=' ', flush=True)
    sma_rows = []
    for i, (f, s, sl, tp) in enumerate(sma_grid):
        if (i + 1) % 100 == 0:
            print(f'{i+1}', end=' ', flush=True)
        r = run_silent(BaselineSmaStrategy, df,
                       {'fast_period': f, 'slow_period': s, 'stop_loss': sl, 'take_profit': tp})
        if r and r['total_trades'] >= 10:
            sma_rows.append(r)

    df_sma = pd.DataFrame(sma_rows).sort_values('net_pl', ascending=False) if sma_rows else pd.DataFrame()
    print(f' -> {len(sma_rows)} geçerli sonuç')

    if not df_sma.empty:
        print(f'\n  SMA TOP 5:')
        for _, row in df_sma.head(5).iterrows():
            print(f'    SMA({int(row["fast_period"])},{int(row["slow_period"])}) '
                  f'SL={row["stop_loss"]:.1%} TP={row["take_profit"]:.1%} | '
                  f'P/L={row["net_pl"]:+,.0f} TL | WR={row["win_rate"]:.0%} | '
                  f'DD={row["max_dd"]:.1f}% | Sharpe={row["sharpe"]:.2f} | '
                  f'Trades={int(row["total_trades"])}')
        best = df_sma.iloc[0].to_dict()
        best['symbol'] = sym
        best['strategy'] = 'SMA_1h'
        all_best.append(best)
        df_sma['symbol'] = sym
        df_sma.to_csv(f'results/opt_1h_{sym.replace(".", "_")}_sma.csv', index=False)

    # === RSI+BB ===
    rsi_grid = list(product(
        [7, 10, 14],
        [30, 35, 40],
        [60, 65, 70],
        [10, 15, 20],
        [1.5, 2.0, 2.5],
        [0.015, 0.02, 0.03],
        [0.03, 0.04, 0.06],
    ))

    print(f'  RSI+BB: {len(rsi_grid)} kombinasyon...', end=' ', flush=True)
    rsi_rows = []
    for i, (rp, ros, rob, bp, bd, sl, tp) in enumerate(rsi_grid):
        if (i + 1) % 200 == 0:
            print(f'{i+1}', end=' ', flush=True)
        r = run_silent(RsiBollingerStrategy, df, {
            'rsi_period': rp, 'rsi_oversold': ros, 'rsi_overbought': rob,
            'bb_period': bp, 'bb_devfactor': bd, 'stop_loss': sl, 'take_profit': tp,
            'cooldown_bars': 3,
        })
        if r and r['total_trades'] >= 10:
            rsi_rows.append(r)

    df_rsi = pd.DataFrame(rsi_rows).sort_values('net_pl', ascending=False) if rsi_rows else pd.DataFrame()
    print(f' -> {len(rsi_rows)} geçerli sonuç')

    if not df_rsi.empty:
        print(f'\n  RSI+BB TOP 5:')
        for _, row in df_rsi.head(5).iterrows():
            print(f'    RSI({int(row["rsi_period"])}) OS={int(row["rsi_oversold"])} '
                  f'OB={int(row["rsi_overbought"])} BB({int(row["bb_period"])},{row["bb_devfactor"]:.1f}) '
                  f'SL={row["stop_loss"]:.1%} TP={row["take_profit"]:.1%} | '
                  f'P/L={row["net_pl"]:+,.0f} TL | WR={row["win_rate"]:.0%} | '
                  f'DD={row["max_dd"]:.1f}% | Sharpe={row["sharpe"]:.2f} | '
                  f'Trades={int(row["total_trades"])}')
        best = df_rsi.iloc[0].to_dict()
        best['symbol'] = sym
        best['strategy'] = 'RSI+BB_1h'
        all_best.append(best)
        df_rsi['symbol'] = sym
        df_rsi.to_csv(f'results/opt_1h_{sym.replace(".", "_")}_rsi.csv', index=False)

# === GENEL ===
print(f'\n{"="*75}')
print(f'  1H SAATLIK — EN IYI SONUCLAR (2.3 yil, 100K TL)')
print(f'{"="*75}')
df_all = pd.DataFrame(all_best)
if not df_all.empty:
    for _, row in df_all.sort_values('net_pl', ascending=False).iterrows():
        print(f'  {row["symbol"]:12s} | {row["strategy"]:10s} | '
              f'P/L={row["net_pl"]:+10,.0f} TL | '
              f'Return={row["return_pct"]:+6.1f}% | '
              f'WR={row["win_rate"]:.0%} | '
              f'DD={row["max_dd"]:.1f}% | '
              f'Sharpe={row["sharpe"]:+.2f} | '
              f'Trades={int(row["total_trades"])}')
    df_all.to_csv('results/best_1h_comparison.csv', index=False)
print(f'{"="*75}')
