# === main.py (v36 – Multi-Exchange & Walk-Forward) ===

import os
import sys
import argparse
import traceback
import datetime as dt

import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config

from utils.data_factory import get_provider
from utils.feature_extractor import add_kmeans_features, build_features
from utils.regime_model import get_regime_labels, analyze_regimes
from analysis.monte_carlo import run_monte_carlo
from ml.meta_labeler import MetaLabeler
from analysis.otr import fit_otr_by_regime

from strategies.ml_strategy import MlStrategy
from strategies.baseline_sma import BaselineSmaStrategy

# -----------------------------------------------------------------
# Path ayarı
# -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '.'))


# -----------------------------------------------------------------
# Equity eğrisi analizer
# -----------------------------------------------------------------
class EquityCurve(bt.Analyzer):
    def start(self):
        self.datetimes = []
        self.values = []
        self.data = self.strategy.data0

    def next(self):
        self.datetimes.append(self.data.datetime.datetime(0))
        self.values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return {"datetimes": self.datetimes, "equity": self.values}


# -----------------------------------------------------------------
# Rejim hattını taşıyabilen PandasData
# -----------------------------------------------------------------
class PandasDataWithRegime(bt.feeds.PandasData):
    lines = ("regime",)
    params = (
        ("datetime", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("openinterest", -1),
        ("regime", -1),
    )


# -----------------------------------------------------------------
# Yüzen (float) boyutlandırma sizer (fallback)
# -----------------------------------------------------------------
class PercentSizerFloat(bt.Sizer):
    params = (("percents", 20),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        price = data.close[0]
        if price <= 0:
            return 0
        available_cash = self.broker.getvalue() * (self.p.percents / 100.0)
        size = available_cash / price
        return size


# -----------------------------------------------------------------
# ATR tabanlı risk sizer — pozisyon başına sabit % risk
# -----------------------------------------------------------------
class ATRRiskSizer(bt.Sizer):
    params = (
        ("risk_pct", 0.005),    # bakiyenin %0.5'i risk/işlem
        ("atr_period", 14),
        ("sl_mult", 2.0),       # SL = atr * sl_mult
        ("fallback_pct", 0.10), # ATR yoksa bakiyenin %10'u
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        price = data.close[0]
        if price <= 0:
            return 0
        equity = self.broker.getvalue()
        risk_amount = equity * self.p.risk_pct

        # ATR'yi veri üzerinden hesapla
        try:
            if len(data) >= self.p.atr_period + 1:
                highs = [data.high[-i] for i in range(self.p.atr_period)]
                lows = [data.low[-i] for i in range(self.p.atr_period)]
                closes = [data.close[-i - 1] for i in range(self.p.atr_period)]
                trs = [max(h - l, abs(h - c), abs(l - c))
                       for h, l, c in zip(highs, lows, closes)]
                atr = sum(trs) / len(trs)
                sl_distance = atr * self.p.sl_mult
                if sl_distance > 0:
                    size = risk_amount / sl_distance
                    return size
        except (IndexError, ZeroDivisionError):
            pass

        # Fallback: bakiyenin sabit yüzdesi
        return (equity * self.p.fallback_pct) / price


# -----------------------------------------------------------------
# Timeframe string → Backtrader TimeFrame + compression
# -----------------------------------------------------------------
def parse_timeframe(tf_str: str):
    """'5m' → (bt.TimeFrame.Minutes, 5), '1h' → (Minutes, 60), '1d' → (Days, 1)"""
    tf_str = tf_str.strip().lower()
    if tf_str.endswith("m"):
        return bt.TimeFrame.Minutes, int(tf_str[:-1])
    elif tf_str.endswith("h"):
        return bt.TimeFrame.Minutes, int(tf_str[:-1]) * 60
    elif tf_str.endswith("d"):
        return bt.TimeFrame.Days, int(tf_str[:-1])
    elif tf_str.endswith("w"):
        return bt.TimeFrame.Weeks, int(tf_str[:-1])
    else:
        return bt.TimeFrame.Minutes, 5  # fallback


# =================================================================
# === BASELINE MODU: sade SMA benchmark ===
# =================================================================
def _extract_metrics(cerebro, strat):
    """Strateji sonuçlarından standart metrik dict'i çıkar."""
    ta = strat.analyzers.trade_analyzer.get_analysis()
    ret = strat.analyzers.returns.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    total_trades = ta.get("total", {}).get("closed", 0) if ta else 0
    win_trades = ta.get("won", {}).get("total", 0) if ta else 0
    winrate = (win_trades / total_trades) if total_trades > 0 else 0.0

    rtot = ret.get("rtot", 0.0) if ret else 0.0
    end_value = float(cerebro.broker.getvalue())
    start_cash = config.START_CASH

    return {
        "symbol": config.SYMBOL,
        "source": getattr(config, "DATA_SOURCE", "binance"),
        "start_cash": start_cash,
        "end_value": end_value,
        "net_pl": end_value - start_cash,
        "rtot": rtot,
        "max_dd": dd.get("max", {}).get("drawdown", 0.0) if dd else 0.0,
        "sharpe": sharpe.get("sharperatio", 0.0) if sharpe else 0.0,
        "sqn": sqn.get("sqn", 0.0) if sqn else 0.0,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "win_rate": winrate,
    }


def run_baseline(optimize: bool = False):
    print("\n=== BASELINE MODU (Sade SMA + SL/TP) ===")

    cerebro = bt.Cerebro(stdstats=not optimize)

    # --- Veri çekme (sadece trade timeframe) ---
    provider = get_provider(config.DATA_SOURCE)
    print(f"\n--- Veri Çekiliyor ({config.TIMEFRAME_TRADE} Trade, kaynak: {provider.name()}) ---")
    df_trade = provider.fetch_ohlcv(
        config.SYMBOL,
        config.TIMEFRAME_TRADE,
        config.TOTAL_BARS_TO_FETCH,
        config.BARS_PER_REQUEST,
    )

    if df_trade is None or df_trade.empty:
        print("Veri çekilemedi veya boş döndü, çıkılıyor.")
        return

    start_date = df_trade.index.min()
    end_date = df_trade.index.max()
    print(f"Sinyal veri aralığı: {start_date} - {end_date}")

    # --- Data feed ---
    bt_tf, bt_comp = parse_timeframe(config.TIMEFRAME_TRADE)
    data_feed_trade = bt.feeds.PandasData(
        dataname=df_trade,
        fromdate=start_date,
        todate=end_date,
        name=config.TIMEFRAME_TRADE,
        timeframe=bt_tf,
        compression=bt_comp,
    )
    cerebro.adddata(data_feed_trade)

    # --- Broker & sizer ---
    cerebro.broker.setcash(config.START_CASH)
    cerebro.broker.setcommission(commission=config.COMMISSION_FEE)
    if config.USE_ATR_POSITION_SIZING:
        cerebro.addsizer(ATRRiskSizer, risk_pct=config.RISK_PER_TRADE)
    else:
        cerebro.addsizer(PercentSizerFloat, percents=config.SIZER_PERCENTS)

    # --- Analizörler ---
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe",
        timeframe=bt_tf,
        compression=bt_comp,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    if not optimize:
        cerebro.addanalyzer(EquityCurve, _name="equity")

    # --- Strateji ekleme / optimizasyon ---
    if optimize:
        print("Optimizasyon modunda çalıştırılıyor (Baseline SMA)...")

        cerebro.optstrategy(
            BaselineSmaStrategy,
            fast_period=config.OPT_TRADE_FAST_SMA,
            slow_period=config.OPT_TRADE_SLOW_SMA,
            stop_loss=config.OPT_STOP_LOSS,
            take_profit=config.OPT_TAKE_PROFIT,
        )

        print("\n--- Optimizasyon Başlatılıyor ---")
        try:
            opt_results = cerebro.run(
                maxcpus=1, optreturn=True, stdstats=False, exactbars=0
            )
            print("\n--- Optimizasyon Tamamlandı ---")

            # Sonuçları CSV'ye yaz
            results_dir = os.path.join(SCRIPT_DIR, "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, "baseline_opt_results.csv")

            rows = []
            total_runs = len(opt_results)
            print(f"Toplam {total_runs} kombinasyon işlenecek...")

            run_counter = 0
            for run_strats in opt_results:
                # optstrategy için run_strats bir liste; ilk eleman strateji
                strat = run_strats[0]
                run_counter += 1
                if run_counter % 10 == 0 or run_counter == total_runs:
                    print(f"   > Deneme {run_counter}/{total_runs} işlendi...")

                params = dict(strat.params._getitems())

                ta = strat.analyzers.trade_analyzer.get_analysis()
                ret = strat.analyzers.returns.get_analysis()
                dd = strat.analyzers.dd.get_analysis()
                sharpe = strat.analyzers.sharpe.get_analysis()
                sqn = strat.analyzers.sqn.get_analysis()

                # Güvenli çekim
                total_trades = ta.get("total", {}).get("closed", 0) if ta else 0
                win_trades = ta.get("won", {}).get("total", 0) if ta else 0
                winrate = (win_trades / total_trades) if total_trades > 0 else 0.0

                rtot = ret.get("rtot", 0.0) if ret else 0.0
                net_profit = rtot * config.START_CASH

                max_dd = dd.get("max", {}).get("drawdown", 0.0) if dd else 0.0
                sharpe_val = sharpe.get("sharperatio", 0.0) if sharpe else 0.0
                sqn_val = sqn.get("sqn", 0.0) if sqn else 0.0

                rows.append(
                    {
                        "fast_period": params.get("fast_period"),
                        "slow_period": params.get("slow_period"),
                        "stop_loss": params.get("stop_loss"),
                        "take_profit": params.get("take_profit"),
                        "net_profit": net_profit,
                        "rtot": rtot,
                        "max_drawdown": max_dd,
                        "sharpe": sharpe_val,
                        "sqn": sqn_val,
                        "trades": total_trades,
                        "winrate": winrate,
                    }
                )

            df_results = pd.DataFrame(rows)
            df_results.to_csv(csv_path, index=False)
            print(f"\nOptimizasyon sonuçları kaydedildi: {csv_path}")
            print(df_results.sort_values("net_profit", ascending=False).head(10))

        except Exception as e:
            print(f"!!! Optimizasyon hatası (baseline): {e}")
            traceback.print_exc()

    else:
        print("Tek çalıştırma (Baseline SMA Benchmark) modu...")
        cerebro.addstrategy(
            BaselineSmaStrategy,
            fast_period=config.BASELINE_TRADE_FAST_SMA,
            slow_period=config.BASELINE_TRADE_SLOW_SMA,
            stop_loss=config.BASELINE_STOP_LOSS,
            take_profit=config.BASELINE_TAKE_PROFIT,
        )

        print(f"\nBaşlangıç Portföy Değeri: {cerebro.broker.getvalue():.2f}")
        results = cerebro.run()
        print(f"Final Portföy Değeri: {cerebro.broker.getvalue():.2f}")

        strat = results[0]
        metrics = _extract_metrics(cerebro, strat)

        eq = strat.analyzers.equity.get_analysis()
        dates = mdates.date2num(eq["datetimes"])
        values = eq["equity"]

        plt.figure(figsize=(10, 4))
        plt.plot_date(dates, values, "-", label="Equity")
        plt.xlabel("Tarih")
        plt.ylabel("Portföy Değeri")
        plt.title(f"Baseline SMA – {config.SYMBOL} Equity Eğrisi")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return metrics


# =================================================================
# === ML MODU: mevcut MlStrategy + K-Means + Meta-Labeler + OTR ===
# =================================================================
def run_ml(optimize: bool = False, full_backtest: bool = False):
    print("\n=== ML MODU (K-Means + MlStrategy) ===")

    cerebro = bt.Cerebro(stdstats=not optimize)

    # --- Veri Çekme ---
    provider = get_provider(config.DATA_SOURCE)
    print(f"\n--- Veri Çekiliyor (kaynak: {provider.name()}) ---")
    trend_bars_to_fetch = (config.TOTAL_BARS_TO_FETCH // 12) + 200

    df_trend = provider.fetch_ohlcv(
        config.SYMBOL,
        config.TIMEFRAME_TREND,
        trend_bars_to_fetch,
        config.BARS_PER_REQUEST,
    )
    if df_trend is None or df_trend.empty:
        print("Trend verisi boş, çıkılıyor.")
        return

    df_trade = provider.fetch_ohlcv(
        config.SYMBOL,
        config.TIMEFRAME_TRADE,
        config.TOTAL_BARS_TO_FETCH,
        config.BARS_PER_REQUEST,
    )
    if df_trade is None or df_trade.empty:
        print("Trade verisi boş, çıkılıyor.")
        return

    # --- K-Means Rejim Analizi ---
    print("\n--- K-Means Rejim Analizi ---")
    df_trend, features = add_kmeans_features(
        df_trend, config.KMEANS_ATR_PERIOD, config.KMEANS_ROC_PERIOD
    )
    model, scaler, labels, features_scaled = get_regime_labels(
        features, config.KMEANS_N_CLUSTERS
    )

    if not optimize:
        analyze_regimes(model, scaler, features_scaled, labels)

    regime_stats = pd.DataFrame(
        features_scaled, columns=["ATR_scaled", "ROC_scaled"]
    )
    regime_stats["regime"] = labels

    # GOOD_REGIME_ID seçimi
    good_regime_to_use = config.GOOD_REGIME_ID

    if getattr(config, "KMEANS_AUTO_SELECT", True) and good_regime_to_use is None:
        good_regime_to_use = int(
            regime_stats.groupby("regime")["ROC_scaled"].mean().idxmax()
        )
        print(
            f"\n[Auto] GOOD_REGIME_ID (En Yüksek ROC) olarak ayarlandı: "
            f"{good_regime_to_use}\n"
        )
    elif good_regime_to_use is None:
        print("\n[Config] GOOD_REGIME_ID = None. K-Means filtresi KAPALI.\n")
    else:
        print(f"\n[Config] GOOD_REGIME_ID kullanılıyor: {good_regime_to_use}\n")

    df_trend["regime"] = labels

    # --- Veri Senkronizasyonu ---
    start_date = df_trend.index.min()
    end_date = df_trend.index.max()
    df_trade = df_trade.loc[start_date:end_date]
    print(f"\nSenkronize veri aralığı: {start_date} - {end_date}")

    # --- Meta-Labeler & OTR Eğitimi (sadece tek çalıştırmada) ---
    meta_model_instance = None
    otr_config_map = {0: (1.2, 2.4), 1: (2.0, 3.5), 2: (1.5, 3.0)}

    if not optimize:
        try:
            X_full = build_features(df_trade, df_trend)
            H = 36  # 3 saatlik (36 * 5m) etiket ufku

            fret = (
                df_trade["close"].pct_change(H).shift(-H).reindex(X_full.index)
            )
            y = (fret > 0).astype(int)
            y.name = "meta_label"

            cut = int(len(X_full) * 0.8)
            X_train, y_train = X_full.iloc[:cut].dropna(), y.iloc[:cut].dropna()
            X_train, y_train = X_train.align(
                y_train, join="inner", axis=0
            )

            if not y_train.empty:
                print(
                    "\n[Diagnostik] Meta-Labeler Eğitim Seti Etiket Dengesi:"
                )
                print(y_train.value_counts(normalize=True).to_string())
            else:
                print(
                    "\n[Diagnostik] Meta-Labeler için eğitim verisi (y_train) "
                    "bulunamadı."
                )

            if not X_train.empty and getattr(
                config, "USE_META_LABELER", False
            ):
                meta_model_instance = MetaLabeler(
                    threshold=config.META_THRESHOLD
                ).fit(X_train, y_train)
            elif getattr(config, "USE_META_LABELER", False):
                print("UYARI: Meta-model eğitimi için veri bulunamadı.")
            else:
                print(
                    "\n[Config] USE_META_LABELER = False. "
                    "Model eğitimi atlanıyor."
                )

            otr_grid = (
                (1.0, 2.0),
                (1.2, 2.4),
                (1.5, 3.0),
                (2.0, 3.5),
                (2.5, 4.0),
            )
            otr_calc_data = {
                "returns": fret.iloc[:cut],
                "atr": X_full["atr_14"].iloc[:cut],
                "regimes": X_full["regime"].iloc[:cut],
            }

            otr_map = fit_otr_by_regime(
                **otr_calc_data,
                grid=otr_grid,
            )
            if len(otr_map) > 0:
                otr_config_map = otr_map
            else:
                print(
                    "UYARI: OTR haritası hesaplanamadı, varsayılan "
                    "(yedek) harita kullanılıyor."
                )

        except Exception as e:
            print(f"!!! Meta-Model / OTR Eğitim Hatası: {e}")
            traceback.print_exc()

    # --- Walk-forward: sadece OOS veri üzerinde backtest ---
    if not optimize and not full_backtest:
        cut_idx = int(len(df_trade) * 0.8)
        oos_start = df_trade.index[cut_idx]
        print(f"\n[Walk-Forward] Backtest sadece OOS verisi üzerinde: {oos_start} - {end_date}")
        print(f"  (Eğitim: ilk %80 = {cut_idx} bar, Test: son %20 = {len(df_trade) - cut_idx} bar)")
        start_date = oos_start
    elif full_backtest and not optimize:
        print("\n[UYARI] --full-backtest: Tüm veri üzerinde çalıştırılıyor (in-sample dahil!)")

    # --- Data feed'ler ---
    bt_tf_trade, bt_comp_trade = parse_timeframe(config.TIMEFRAME_TRADE)
    bt_tf_trend, bt_comp_trend = parse_timeframe(config.TIMEFRAME_TREND)

    data_feed_trade = bt.feeds.PandasData(
        dataname=df_trade,
        fromdate=start_date,
        todate=end_date,
        name=config.TIMEFRAME_TRADE,
        timeframe=bt_tf_trade,
        compression=bt_comp_trade,
    )
    cerebro.adddata(data_feed_trade)

    data_feed_trend = PandasDataWithRegime(
        dataname=df_trend,
        fromdate=start_date,
        todate=end_date,
        name=config.TIMEFRAME_TREND,
        timeframe=bt_tf_trend,
        compression=bt_comp_trend,
    )
    cerebro.adddata(data_feed_trend)

    # --- Broker & sizer ---
    cerebro.broker.setcash(config.START_CASH)
    cerebro.broker.setcommission(commission=config.COMMISSION_FEE)
    if config.USE_ATR_POSITION_SIZING:
        cerebro.addsizer(ATRRiskSizer, risk_pct=config.RISK_PER_TRADE)
    else:
        cerebro.addsizer(PercentSizerFloat, percents=config.SIZER_PERCENTS)

    # --- Analizörler ---
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe",
        timeframe=bt_tf_trade,
        compression=bt_comp_trade,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    if not optimize:
        cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
        cerebro.addanalyzer(EquityCurve, _name="equity")

    # --- Strateji / Optimizasyon ---
    if optimize:
        print("Optimizasyon modunda çalıştırılıyor (MlStrategy)...")
        cerebro.optstrategy(
            MlStrategy,
            trade_fast_sma=[8, 10, 12, 14],
            trade_slow_sma=[56, 58, 60, 62, 64],
            stop_loss=[0.018, 0.02, 0.022],
            take_profit=[0.045, 0.05, 0.055],
            good_regime_id=good_regime_to_use,
            printlog=False,
            use_meta=False,
            use_dynamic_risk=False,
        )

        try:
            opt_results = cerebro.run(
                maxcpus=1, optreturn=True, stdstats=False, exactbars=0
            )
            print("\n--- ML Optimizasyon Tamamlandı ---")
            # Burada istersen benzer şekilde CSV’ye loglayabilirsin.
        except Exception as e:
            print(f"!!! ML Optimizasyon hatası: {e}")
            traceback.print_exc()

    else:
        print("Tek çalıştırma modunda (Meta-Labeler ve OTR ile)...")
        cerebro.addstrategy(
            MlStrategy,
            trade_fast_sma=config.TRADE_FAST_SMA,
            trade_slow_sma=config.TRADE_SLOW_SMA,
            trend_fast_sma=config.TREND_FAST_SMA,
            trend_slow_sma=config.TREND_SLOW_SMA,
            stop_loss=config.STOP_LOSS,
            take_profit=config.TAKE_PROFIT,
            use_atr_stops=getattr(config, "USE_ATR_STOPS", True),
            atr_period=getattr(config, "ATR_PERIOD", 14),
            atr_sl_mult=getattr(config, "ATR_SL_MULT", 2.0),
            atr_tp_mult=getattr(config, "ATR_TP_MULT", 3.5),
            min_cross_strength=getattr(
                config, "MIN_CROSS_STRENGTH", 0.25
            ),
            cooldown_bars=getattr(config, "COOL_DOWN_BARS", 24),
            adx_min=getattr(config, "ADX_MIN", 16),
            good_regime_id=good_regime_to_use,
            arima_enabled=config.ARIMA_ENABLED,
            arima_order=config.ARIMA_ORDER,
            arima_lookback=config.ARIMA_LOOKBACK,
            arima_forecast_steps=config.ARIMA_FORECAST_STEPS,
            printlog=True,
            allow_short=True,
            use_meta=getattr(config, "USE_META_LABELER", False),
            meta_model=meta_model_instance,
            bet_sizing=getattr(config, "USE_BET_SIZING", False),
            use_dynamic_risk=getattr(config, "USE_DYNAMIC_RISK", True),
            regime_sl_tp=otr_config_map,
            filter_trading_hours=getattr(config, "FILTER_TRADING_HOURS", True),
        )

        print(f"\nBaşlangıç Portföy Değeri: {cerebro.broker.getvalue():.2f}")
        results = cerebro.run()
        print(f"Final Portföy Değeri: {cerebro.broker.getvalue():.2f}")

        strat = results[0]
        return _extract_metrics(cerebro, strat)


# =================================================================
# === CLI Giriş Noktası ===
# =================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AlgoTrade Bot – Baseline & ML Backtest/Optimizasyon"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "ml"],
        help="Çalışma modu: 'baseline' (sade SMA) veya 'ml' (ML stratejisi)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimizasyon modunda çalıştır",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Sembol override (ör: BTC/USDT, THYAO.IS)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["binance", "yfinance"],
        help="Veri kaynağı override",
    )
    parser.add_argument(
        "--full-backtest",
        action="store_true",
        help="ML modunda tüm veri üzerinde backtest (in-sample dahil)",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="config.SYMBOLS listesindeki tüm sembolleri sırayla çalıştır",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Walk-forward optimization (rolling window OOS test)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading modu (sanal para ile canlı sinyal)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Canlı trading modu (gerçek emir — Binance)",
    )

    args = parser.parse_args()

    # CLI override'ları config'e uygula
    if args.symbol:
        config.SYMBOL = args.symbol
    if args.source:
        config.DATA_SOURCE = args.source

    # Walk-Forward Optimization
    if args.walk_forward:
        from utils.walk_forward import run_walk_forward

        provider = get_provider(config.DATA_SOURCE)
        print(f"\n--- Walk-Forward: Veri çekiliyor ({config.SYMBOL}) ---")
        df_trade = provider.fetch_ohlcv(
            config.SYMBOL, config.TIMEFRAME_TRADE,
            config.TOTAL_BARS_TO_FETCH, config.BARS_PER_REQUEST,
        )
        if df_trade is None or df_trade.empty:
            print("Veri çekilemedi!")
            sys.exit(1)

        bt_tf, bt_comp = parse_timeframe(config.TIMEFRAME_TRADE)

        param_grid = {
            "fast_period": config.OPT_TRADE_FAST_SMA,
            "slow_period": config.OPT_TRADE_SLOW_SMA,
            "stop_loss": config.OPT_STOP_LOSS,
            "take_profit": config.OPT_TAKE_PROFIT,
        }

        run_walk_forward(
            strategy_cls=BaselineSmaStrategy,
            df_trade=df_trade,
            param_grid=param_grid,
            train_bars=50000,
            test_bars=10000,
            bt_tf=bt_tf,
            bt_comp=bt_comp,
            optimize_metric="sharpe",
        )
        sys.exit(0)

    # Live/Paper trading modu
    if args.paper or args.live:
        from live.engine import TradingEngine
        from live.signal_generator import SignalGenerator
        from live.paper_broker import PaperBroker

        # yfinance ile live trading engellenmiş
        if args.live and getattr(config, "DATA_SOURCE", "binance") == "yfinance":
            print("HATA: yfinance sembollerinde canlı trading yapılamaz. --paper kullanın.")
            sys.exit(1)

        # Signal generator
        sig_gen = SignalGenerator(
            trade_fast_sma=config.TRADE_FAST_SMA,
            trade_slow_sma=config.TRADE_SLOW_SMA,
            trend_fast_sma=config.TREND_FAST_SMA,
            trend_slow_sma=config.TREND_SLOW_SMA,
            adx_min=getattr(config, "ADX_MIN", 20),
            atr_period=getattr(config, "ATR_PERIOD", 14),
            min_cross_strength=getattr(config, "MIN_CROSS_STRENGTH", 0.25),
            cooldown_bars=getattr(config, "COOL_DOWN_BARS", 12),
            stop_loss=config.STOP_LOSS,
            take_profit=config.TAKE_PROFIT,
            allow_short=True,
            filter_trading_hours=getattr(config, "FILTER_TRADING_HOURS", True),
        )

        # Broker
        if args.live:
            from live.live_broker import LiveBroker
            broker = LiveBroker(sandbox=True)  # varsayılan testnet
        else:
            broker = PaperBroker(cash=config.START_CASH, commission=config.COMMISSION_FEE)

        # Provider
        provider = get_provider(config.DATA_SOURCE)

        # State dosya yolu
        safe_symbol = config.SYMBOL.replace("/", "_").replace(".", "_")
        mode_tag = "live" if args.live else "paper"
        state_path = os.path.join(
            getattr(config, "LIVE_STATE_DIR", "data"),
            f"state_{safe_symbol}_{mode_tag}.json",
        )

        engine = TradingEngine(
            broker=broker,
            signal_gen=sig_gen,
            provider=provider,
            symbol=config.SYMBOL,
            timeframe_trade=config.TIMEFRAME_TRADE,
            timeframe_trend=config.TIMEFRAME_TREND,
            state_path=state_path,
            poll_interval=getattr(config, "LIVE_POLL_INTERVAL", 60),
            warmup_bars=getattr(config, "LIVE_WARMUP_BARS", 300),
            risk_per_trade=config.RISK_PER_TRADE,
        )
        engine.run()
        sys.exit(0)

    # Multi-symbol modu
    if args.multi:
        from runner import run_multi_symbol
        run_multi_symbol(
            mode=args.mode,
            optimize=args.optimize,
            full_backtest=args.full_backtest,
        )
    elif args.mode == "baseline":
        run_baseline(optimize=args.optimize)
    else:
        run_ml(optimize=args.optimize, full_backtest=args.full_backtest)


if __name__ == "__main__":
    main()
