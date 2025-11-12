# === GÜNCELLENDİ: main.py (v34 - Diagnostik ve Config Okuma) ===
import backtrader as bt
import datetime
import pandas as pd
import argparse
import traceback
import sys # Path eklemek için
import os  # Path eklemek için

# --- YENİ: Yeni modüllerin yolunu (path) ekle ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '.'))
# ---------------------------------------------

import config
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from utils.data_fetcher import fetch_binance_data
from utils.feature_extractor import add_kmeans_features, build_features 
from utils.regime_model import get_regime_labels, analyze_regimes
from analysis.monte_carlo import run_monte_carlo
from strategies.ml_strategy import MlStrategy 

# --- YENİ: Meta-Labeler ve OTR importları ---
from ml.meta_labeler import MetaLabeler
from analysis.otr import fit_otr_by_regime
# ---------------------------------------------


# === Equity eğrisi için Analyzer === (Değişiklik yok)
class EquityCurve(bt.Analyzer):
    def start(self): self.datetimes = []; self.values = []; self.data = self.strategy.data0 
    def next(self): self.datetimes.append(self.data.datetime.datetime(0)); self.values.append(self.strategy.broker.getvalue())
    def get_analysis(self): return {'datetimes': self.datetimes, 'equity': self.values}
# =================================================================

# === 'regime' hattını okuyabilen özel PandasData sınıfı === (Değişiklik yok)
class PandasDataWithRegime(bt.feeds.PandasData):
    lines = ('regime',)
    params = (('datetime', None), ('open', -1), ('high', -1), ('low', -1), ('close', -1), ('volume', -1), ('openinterest', -1), ('regime', -1),)
# ======================================================================

# === Yüzen (float) boyutlandırma Sizer'ı === (Değişiklik yok)
class PercentSizerFloat(bt.Sizer):
    params = (('percents', 95),) 
    def _getsizing(self, comminfo, cash, data, isbuy):
        price = data.close[0];
        if price <= 0: return 0
        available_cash = self.broker.getvalue() * (self.p.percents / 100.0)
        size = available_cash / price
        return size
# =====================================================

def run_backtest(optimize=False):
    cerebro = bt.Cerebro(stdstats=not optimize) 

    # --- Veri Çekme --- (Değişiklik yok)
    print("\n--- Veri Çekiliyor (1h Trend & 5m Trade) ---")
    trend_bars_to_fetch = (config.TOTAL_BARS_TO_FETCH // 12) + 200 
    df_trend = fetch_binance_data(config.SYMBOL, config.TIMEFRAME_TREND, trend_bars_to_fetch, config.BARS_PER_REQUEST)
    if (df_trend is None or df_trend.empty): return
    df_trade = fetch_binance_data(config.SYMBOL, config.TIMEFRAME_TRADE, config.TOTAL_BARS_TO_FETCH, config.BARS_PER_REQUEST)
    if (df_trade is None or df_trade.empty): return

    # --- K-Means Rejim Analizi ---
    print("\n--- K-Means Rejim Analizi ---")
    df_trend, features = add_kmeans_features(df_trend, config.KMEANS_ATR_PERIOD, config.KMEANS_ROC_PERIOD)
    model, scaler, labels, features_scaled = get_regime_labels(features, config.KMEANS_N_CLUSTERS)
    if not optimize: analyze_regimes(model, scaler, features_scaled, labels)
    regime_stats = pd.DataFrame(features_scaled, columns=['ATR_scaled','ROC_scaled']); regime_stats['regime'] = labels
    
    # === YENİ MANTIK (v34) ===
    # 'GOOD_REGIME_ID'yi config'den al veya otomatize et
    good_regime_to_use = config.GOOD_REGIME_ID
    # KMEANS_AUTO_SELECT = True ise ve ID None ise, en iyi ROC'u seç
    if getattr(config, "KMEANS_AUTO_SELECT", True) and good_regime_to_use is None:
        good_regime_to_use = int(regime_stats.groupby('regime')['ROC_scaled'].mean().idxmax())
        print(f"\n[Auto] GOOD_REGIME_ID (En Yüksek ROC) olarak ayarlandı: {good_regime_to_use}\n")
    # KMEANS_AUTO_SELECT = False ise ve ID None ise, filtre kapalı kalır
    elif good_regime_to_use is None:
        print(f"\n[Config] GOOD_REGIME_ID = None. K-Means filtresi KAPALI.\n")
    else:
         print(f"\n[Config] GOOD_REGIME_ID kullanılıyor: {good_regime_to_use}\n")
    # === YENİ MANTIK BİTİŞ ===
         
    df_trend['regime'] = labels

    # --- Veri Hizalama (Senkronizasyon) ---
    start_date = df_trend.index.min(); end_date = df_trend.index.max()   
    df_trade = df_trade.loc[start_date:end_date]
    print(f"\nSenkronize veri aralığı: {start_date} - {end_date}")
    
    # --- YENİ: Meta-Labeler ve OTR Eğitimi (Sadece tekli çalıştırmada) ---
    meta_model_instance = None
    otr_config_map = {0:(1.2,2.4), 1:(2.0,3.5), 2:(1.5,3.0)} # Varsayılan/Yedek
    
    if not optimize:
        try:
            X_full = build_features(df_trade, df_trend)
            H = 36 # 3 saatlik (36 * 5m) etiket ufku
            fret = df_trade['close'].pct_change(H).shift(-H).reindex(X_full.index)
            y = (fret > 0).astype(int)
            y.name = 'meta_label'
            
            cut = int(len(X_full) * 0.8)
            X_train, y_train = X_full.iloc[:cut].dropna(), y.iloc[:cut].dropna()
            X_train, y_train = X_train.align(y_train, join='inner', axis=0)
            
            # === YENİ (v34): ETİKET DENGESİNİ KONTROL ET ===
            if not y_train.empty:
                print("\n[Diagnostik] Meta-Labeler Eğitim Seti Etiket Dengesi:")
                print(y_train.value_counts(normalize=True).to_string())
            else:
                print("\n[Diagnostik] Meta-Labeler için eğitim verisi (y_train) bulunamadı.")
            # ===============================================

            if not X_train.empty and getattr(config, "USE_META_LABELER", False):
                meta_model_instance = MetaLabeler(threshold=0.55).fit(X_train, y_train)
            elif getattr(config, "USE_META_LABELER", False):
                print("UYARI: Meta-model eğitimi için veri bulunamadı.")
            else:
                print("\n[Config] USE_META_LABELER = False. Model eğitimi atlanıyor.")
                
            otr_grid = ((1.0, 2.0),(1.2, 2.4),(1.5, 3.0),(2.0, 3.5),(2.5, 4.0))
            
            otr_calc_data = {
                'returns': fret.iloc[:cut],
                'atr': X_full['atr_14'].iloc[:cut], 
                'regimes': X_full['regime'].iloc[:cut] 
            }
            otr_map = fit_otr_by_regime(**otr_calc_data, grid=otr_grid)
            
            if len(otr_map) > 0:
                otr_config_map = otr_map 
            else:
                print("UYARI: OTR haritası hesaplanamadı, varsayılan (yedek) harita kullanılıyor.")
                
        except Exception as e:
            print(f"!!! Meta-Model / OTR Eğitim Hatası: {e}")
            traceback.print_exc()
    # --- YENİ EĞİTİM BLOĞU SONU ---


    if optimize:
        print("Optimizasyon modunda çalıştırılıyor...")
        cerebro.optstrategy(MlStrategy,
            trade_fast_sma=[8, 10, 12, 14], trade_slow_sma=[56, 58, 60, 62, 64], 
            stop_loss=[0.018, 0.02, 0.022], take_profit=[0.045, 0.05, 0.055],
            good_regime_id=good_regime_to_use, # <-- Düzeltildi v34
            printlog=False,
            use_meta=False, use_dynamic_risk=False,
        )
    else:
        print("Tek çalıştırma modunda (Meta-Labeler ve OTR ile) çalıştırılıyor...")
        cerebro.addstrategy(MlStrategy, 
            # Ana Strateji
            trade_fast_sma=config.TRADE_FAST_SMA, 
            trade_slow_sma=config.TRADE_SLOW_SMA,
            trend_fast_sma=config.TREND_FAST_SMA, 
            trend_slow_sma=config.TREND_SLOW_SMA,
            
            # Risk (Yedek - OTR veya ATR kapalıysa)
            stop_loss=config.STOP_LOSS, 
            take_profit=config.TAKE_PROFIT,
            
            # Risk (ATR)
            use_atr_stops=getattr(config, "USE_ATR_STOPS", True),
            atr_period=getattr(config, "ATR_PERIOD", 14),
            atr_sl_mult=getattr(config, "ATR_SL_MULT", 2.0),
            atr_tp_mult=getattr(config, "ATR_TP_MULT", 3.5),
            
            # Filtreler (Config'den okunacak)
            min_cross_strength=getattr(config, "MIN_CROSS_STRENGTH", 0.25),
            cooldown_bars=getattr(config, "COOL_DOWN_BARS", 24),
            adx_min=getattr(config, "ADX_MIN", 16),
            
            # Rejim
            good_regime_id=good_regime_to_use, # <-- Düzeltildi v34
            
            # ARIMA
            arima_enabled=config.ARIMA_ENABLED, 
            arima_order=config.ARIMA_ORDER,
            arima_lookback=config.ARIMA_LOOKBACK, 
            arima_forecast_steps=config.ARIMA_FORECAST_STEPS,
            
            # Diğer
            printlog=True, 
            allow_short=True,
            
            # === YENİ (v34): ML Ayarları Config'den okunuyor ===
            use_meta=getattr(config, "USE_META_LABELER", False),
            meta_model=meta_model_instance,
            bet_sizing=getattr(config, "USE_BET_SIZING", False),
            use_dynamic_risk=getattr(config, "USE_DYNAMIC_RISK", True),
            regime_sl_tp=otr_config_map
            # ============================================
        )

    # --- Veri Ekleme --- (Değişiklik yok)
    data_feed_trade = bt.feeds.PandasData(dataname=df_trade, fromdate=start_date, todate=end_date, name=config.TIMEFRAME_TRADE, timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.adddata(data_feed_trade)
    data_feed_trend = PandasDataWithRegime(dataname=df_trend, fromdate=start_date, todate=end_date, name=config.TIMEFRAME_TREND, timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.adddata(data_feed_trend) 

    # --- Broker ve Sizer --- (Değişiklik yok)
    cerebro.broker.setcash(config.START_CASH)
    cerebro.broker.setcommission(commission=config.COMMISSION_FEE)
    cerebro.addsizer(PercentSizerFloat, percents=config.SIZER_PERCENTS)
    
    # --- Analizörler --- (Değişiklik yok)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe', timeframe=bt.TimeFrame.Minutes, compression=5, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    if not optimize:
        cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
        cerebro.addanalyzer(EquityCurve, _name='equity')

    # --- Backtest/Optimizasyon ---
    if optimize:
        # (Optimizasyon kısmı - Değişiklik yok)
        print(f"\n--- Optimizasyon Başlatılıyor (Toplam {len(cerebro.strats)} koşu) ---")
        try:
            opt_results = cerebro.run(maxcpus=1, optreturn=True, stdstats=False, exactbars=0)
            print("\n--- Optimizasyon Tamamlandı ---")
            final_results_list = []
            print("\n--- Optimizasyon Sonuçları İşleniyor ---"); run_counter = 0
            for run_result_list in opt_results:
                run_counter += 1
                try:
                    if not run_result_list: continue
                    strategy_instance = run_result_list[0]
                    net_profit = (strategy_instance.broker.getvalue() - config.START_CASH)
                    ta, dd, sh, sqn = (strategy_instance.analyzers.trade_analyzer.get_analysis() or {}, strategy_instance.analyzers.dd.get_analysis() or {},
                                       strategy_instance.analyzers.sharpe.get_analysis() or {}, strategy_instance.analyzers.sqn.get_analysis() or {})
                    total_trades = (ta.get('total') or {}).get('total') or 0; won_trades   = (ta.get('won') or {}).get('total') or 0
                    winrate = (won_trades / total_trades * 100.0) if total_trades else 0.0
                    won_pnl  = (ta.get('won') or {}).get('pnl', {}).get('total', 0.0) or 0.0; lost_pnl = (ta.get('lost') or {}).get('pnl', {}).get('total', 0.0) or 0.0
                    pf = (won_pnl / abs(lost_pnl)) if lost_pnl != 0 else None
                    maxdd = (dd.get('max') or {}).get('drawdown', None); sharpe_ratio = sh.get('sharperatio', None); sqn_val = sqn.get('sqn', None)
                    result_dict = {'FastSMA': strategy_instance.params.trade_fast_sma, 'SlowSMA': strategy_instance.params.trade_slow_sma,
                                   'StopLoss': strategy_instance.params.stop_loss, 'TakeProfit': strategy_instance.params.take_profit,
                                   'NetProfit': round(net_profit, 2), 'MaxDD_%': round(maxdd, 2) if maxdd is not None else None,
                                   'Sharpe': round(sharpe_ratio, 3) if sharpe_ratio is not None else None, 'PF': round(pf, 2) if pf is not None else None, 
                                   'WinRate_%': round(winrate, 2), 'Trades': int(total_trades), 'SQN': round(sqn_val, 2) if sqn_val is not None else None,}
                    final_results_list.append(result_dict)
                except Exception as loop_error: print(f"!!! Çalıştırma #{run_counter} hatası: {loop_error}"); traceback.print_exc()
            print(f"\nToplam {run_counter} çalıştırma işlendi."); results_df = pd.DataFrame(final_results_list)
            if not results_df.empty:
                 filt = results_df.copy(); filt = filt[(filt['MaxDD_%'].notna()) & (filt['Sharpe'].notna()) & (filt['Trades'] >= 10) & (filt['MaxDD_%'] <= 35) & (filt['Sharpe'] >= 0.5)]
                 show_df = (filt if not filt.empty else results_df).sort_values(by=['NetProfit', 'MaxDD_%'], ascending=[False, True])
                 cols = ['FastSMA','SlowSMA','StopLoss','TakeProfit','NetProfit','MaxDD_%','Sharpe','PF','WinRate_%','Trades','SQN']
                 print("\n--- En İyi 5 ---"); print(show_df[cols].head(5).to_string(index=False))
                 results_df.to_csv('opt_results_full.csv', index=False); show_df.head(5).to_csv('opt_results_top5.csv', index=False)
                 print("\nDosyalar yazıldı: opt_results_full.csv, opt_results_top5.csv")
                 def run_single_report(fast, slow, sl, tp, tag):
                     print(f"  -> Top {tag} raporu/grafiği...")
                     cerebro2 = bt.Cerebro(stdstats=False)
                     data_feed_trade_rep = bt.feeds.PandasData(dataname=df_trade, fromdate=start_date, todate=end_date, name=config.TIMEFRAME_TRADE, timeframe=bt.TimeFrame.Minutes, compression=5)
                     data_feed_trend_rep = PandasDataWithRegime(dataname=df_trend, fromdate=start_date, todate=end_date, name=config.TIMEFRAME_TREND, timeframe=bt.TimeFrame.Minutes, compression=60)
                     cerebro2.adddata(data_feed_trade_rep); cerebro2.adddata(data_feed_trend_rep)
                     cerebro2.broker.setcash(config.START_CASH); cerebro2.broker.setcommission(commission=config.COMMISSION_FEE)
                     cerebro2.addsizer(PercentSizerFloat, percents=config.SIZER_PERCENTS); cerebro2.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer'); cerebro2.addanalyzer(bt.analyzers.DrawDown, _name='dd')
                     cerebro2.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe', timeframe=bt.TimeFrame.Minutes, compression=5, riskfreerate=0.0)
                     cerebro2.addanalyzer(bt.analyzers.SQN, _name='sqn'); cerebro2.addanalyzer(EquityCurve, _name='equity') 
                     cerebro2.addstrategy(MlStrategy, 
                         trade_fast_sma=fast, trade_slow_sma=slow, trend_fast_sma=config.TREND_FAST_SMA, trend_slow_sma=config.TREND_SLOW_SMA,
                         stop_loss=sl, take_profit=tp, good_regime_id=good_regime_to_use, # <-- Düzeltildi v34
                         **{k: v for k, v in config.__dict__.items() if not k.startswith('__')},
                         use_meta=False, use_dynamic_risk=False, 
                         printlog=False, allow_short=True
                     )
                     res = cerebro2.run(); st = res[0]; final_val = cerebro2.broker.getvalue(); net_p = final_val - config.START_CASH
                     ta, dd, sh, sq = (st.analyzers.trade_analyzer.get_analysis() or {}, st.analyzers.dd.get_analysis() or {}, st.analyzers.sharpe.get_analysis() or {}, st.analyzers.sqn.get_analysis() or {})
                     analysis = st.analyzers.equity.get_analysis(); eq_dts = analysis.get('datetimes', []); eq_vals = analysis.get('equity', [])
                     with open(f"report_top{tag}.txt", "w", encoding="utf-8") as f:
                         f.write(f"[TOP {tag}] fast={fast} slow={slow} SL={sl} TP={tp}\n"); f.write(f"Final: {final_val:,.2f} Net: {net_p:,.2f} (%{(net_p/config.START_CASH)*100:.2f})\n")
                         f.write(f"Sharpe: {sh.get('sharperatio', 'N/A')}\n"); f.write(f"MaxDD %: {(dd.get('max') or {}).get('drawdown', 'N/A')}\n")
                         tot=(ta.get('total')or{}).get('total',0); won=(ta.get('won')or{}).get('total',0); los=(ta.get('lost')or{}).get('total',0); winrate_rep = (won/tot*100.0) if tot else 0.0; f.write(f"Trades:{tot} Win:{won}({winrate_rep:.2f}%) Loss:{los}\n")
                         won_pnl_rep=(ta.get('won')or{}).get('pnl',{}).get('total',0.0)or 0.0; lost_pnl_rep=(ta.get('lost')or{}).get('pnl',{}).get('total',0.0)or 0.0
                         pf_rep=(won_pnl_rep/abs(lost_pnl_rep)) if lost_pnl_rep!=0 else None; f.write(f"PF: {pf_rep}\n"); f.write(f"SQN: {sq.get('sqn', 'N/A')}\n")
                     try:
                         if not eq_dts: return 
                         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                         ax1.plot(eq_dts, eq_vals); ax1.set_title(f'Equity Curve (TOP {tag} - Final: {final_val:,.2f})'); ax1.grid(True, alpha=0.3)
                         ax2.plot(df_trade.index, df_trade['close'], color='gray', alpha=0.7)
                         ev = getattr(st, 'plot_events', []);
                         if ev:
                             def sel(tag=None, side=None): return [(d,p) for (d,p,s,t) in ev if (tag is None or t==tag) and (side is None or s==side)]
                             buy_ent=sel(tag='ENTRY',side='BUY'); sell_ent=sel(tag='ENTRY_SHORT',side='SELL'); tp_pts=sel(tag='TP'); sl_pts=sel(tag='SL')
                             def scatter(points, marker, label, color): 
                                 if points: xs, ys = zip(*points); ax2.scatter(xs, ys, marker=marker, s=60, alpha=0.9, label=label, color=color, edgecolors='k', linewidths=0.5) 
                             scatter(buy_ent, '^', 'Buy Entry', 'lime'); scatter(sell_ent,'v', 'Short Entry', 'red'); scatter(tp_pts,  'o', 'Take Profit', 'blue'); scatter(sl_pts,  'x', 'Stop Loss', 'magenta'); ax2.legend(loc='upper left')
                         
                         # === PYLANCE HATA DÜZELTMESİ (v32) ===
                         ax2.set_title(f'Price (BTC/USDT 5m)'); ax2.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f"equity_top{tag}.png", dpi=150); plt.close(fig)
                         # ==================================
                     
                     except Exception as e: print(f"  [Grafik Hatası] Top {tag}: {e}")
                 print("\n--- En İyi 5 Sistem Rapor/Grafik Oluşturma ---")
                 top5 = show_df.head(5)
                 for rank, (_, r) in enumerate(top5.iterrows(), start=1):
                     run_single_report(fast=int(r['FastSMA']), slow=int(r['SlowSMA']), sl=float(r['StopLoss']), tp=float(r['TakeProfit']), tag=rank)
                 print("\nTop 5 rapor/grafik tamamlandı.")
            else: print("\n--- Optimizasyon Sonuçları --- \nHiçbir sonuç işlenemedi.")
        except Exception as e: print(f"\n!!! Optimizasyon hatası: {e}"); traceback.print_exc()

    else: # --- Tek Çalıştırma Modu ---
        print("\n--- Backtest Başlatılıyor (Tek Çalıştırma - Meta/OTR Aktif) ---")
        results = cerebro.run()
        strategy_instance = results[0]
        print("\n--- Backtest Tamamlandı ---")
        final_value = cerebro.broker.getvalue(); net_profit = final_value - config.START_CASH
        print(f"\n--- SONUÇ RAPORU (Tek Çalıştırma) ---")
        print(f"Başlangıç: {config.START_CASH:,.2f} Bitiş: {final_value:,.2f} Net: {net_profit:,.2f} (%{(net_profit/config.START_CASH)*100:.2f}%)")
        print("\n--- Detaylı Analizörler ---")
        
        # === HATA DÜZELTME v29 BAŞLANGIÇ ===
        trade_analyzer = strategy_instance.analyzers.trade_analyzer
        ta = trade_analyzer.get_analysis() 
        dd, sh, sqn = (strategy_instance.analyzers.dd.get_analysis(), 
                       strategy_instance.analyzers.sharpe.get_analysis(), 
                       strategy_instance.analyzers.sqn.get_analysis())
        # === HATA DÜZELTME v29 BİTİŞ ===

        
        # === HATA DÜZELTME BLOĞU (v26) ===
        sharpe_val = sh.get('sharperatio', None) 
        sharpe_str = f'{sharpe_val:.3f}' if sharpe_val is not None else 'N/A'
        print(f"Sharpe (Ann.): {sharpe_str}")

        dd_val = dd.get('max', {}).get('drawdown', None)
        dd_str = f'{dd_val:.2f}%' if dd_val is not None else 'N/A'
        print(f"MaxDD %: {dd_str}")
        
        sqn_val = sqn.get('sqn', None)
        sqn_str = f'{sqn_val:.2f}' if sqn_val is not None else 'N/A'
        print(f"SQN: {sqn_str}")
        # === HATA DÜZELTME BLOĞU BİTİŞ ===

        if ta and ta.get('total', {}).get('total', 0) > 0:
            print("\n--- İşlem Analizi ---"); print(f"Trades:{ta.total.total} Win:{ta.won.total} Loss:{ta.lost.total}")
            print(f"Win Rate: {(ta.won.total/ta.total.total)*100:.2f}%")
            if ta.won.total>0: print(f"Avg Win: {ta.won.pnl.average:.2f}"); 
            if ta.lost.total>0: print(f"Avg Loss: {ta.lost.pnl.average:.2f}")
            if ta.lost.pnl.total!=0: print(f"PF: {abs(ta.won.pnl.total/ta.lost.pnl.total):.2f}")
        pnl_list = strategy_instance.closed_trades_pnl; 
        if pnl_list: run_monte_carlo(pnl_list, n_simulations=5000) 
        else: print("\nMonte Carlo için işlem yok.")
        print("\nGrafik çizdiriliyor...")
        
        # === ANA GRAFİK ÇİZİMİ (v27 - Tüm çıkışlar) ===
        try:
            analysis = strategy_instance.analyzers.equity.get_analysis(); eq_dts = analysis.get('datetimes', []); eq_vals = analysis.get('equity', [])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            ax1.plot(eq_dts, eq_vals); ax1.set_title(f'Equity Curve (Tek Çalıştırma) - Final: {final_value:,.2f}'); ax1.grid(True, alpha=0.3)
            ax2.plot(df_trade.index, df_trade['close'], color='gray', alpha=0.7)
            
            all_events = getattr(strategy_instance, 'plot_events', [])
            
            def scatter(points, marker, label, color, ax): 
                if points: 
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, marker=marker, s=60, alpha=0.9, label=label, color=color, edgecolors='k', linewidths=0.5)
            
            if all_events:
                def sel(events, tag=None, side=None): 
                    return [(d,p) for (d,p,s,t) in events if (tag is None or t==tag) and (side is None or s==side)]
                
                buy_ent=sel(all_events, tag='ENTRY',side='BUY'); sell_ent=sel(all_events, tag='ENTRY_SHORT',side='SELL')
                tp_pts=sel(all_events, tag='TP'); sl_pts=sel(all_events, tag='SL')
                exit_long=sel(all_events, tag='EXIT', side='SELL'); exit_short=sel(all_events, tag='EXIT_SHORT', side='BUY')
                time_stop_pts = sel(all_events, tag='TIME_STOP')

                scatter(buy_ent, '^', 'Buy Entry', 'lime', ax2)
                scatter(sell_ent,'v', 'Short Entry', 'red', ax2)
                scatter(tp_pts,  'o', 'Take Profit', 'blue', ax2)
                scatter(sl_pts,  'x', 'Stop Loss', 'magenta', ax2)
                scatter(exit_long, 'v', 'Strategy Exit', 'orange', ax2) 
                scatter(exit_short, '^', 'Strategy Exit', 'orange', ax2)
                scatter(time_stop_pts, 's', 'Time Stop', 'cyan', ax2)

                ax2.legend(loc='upper left')
                
            ax2.set_title(f'Price (BTC/USDT 5m)'); ax2.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f"report_single_run_equity.png", dpi=150); plt.close(fig)
            print("Grafik 'report_single_run_equity.png' olarak kaydedildi.")
        except Exception as e: print(f"  [Ana Grafik Hatası] Tek çalıştırma grafiği: {e}")
        # === ANA GRAFİK ÇİZİMİ BİTİŞ ===
        
        
        # === YENİ: PARÇA PARÇA (ZOOM-IN) İŞLEM GRAFİKLERİ (v33 - Sizin yamanızla düzeltildi) ===
        print("\n'Parça Parça' (Zoom-in) işlem grafikleri oluşturuluyor...")
        try:
            # DÜZELTME (v31): Veriyi doğrudan stratejiden al (kullanıcının önerisi)
            trades_list = getattr(strategy_instance, 'trade_history', [])
            
            if not trades_list:
                print("Çizilecek işlem bulunamadı (trade_history boş).")
            
            else:
                trades_to_plot = trades_list[-10:] # Son 10 kapalı işlemi al
                print(f"Toplam {len(trades_list)} kapalı işlem bulundu. Son {len(trades_to_plot)} tanesi çiziliyor...")
                
                # --- Yardımcı scatter fonksiyonu (v33 Düzeltmesi) ---
                def scatter(dt, px, label, color, ax, marker='o'):
                    ax.scatter(dt, px, marker=marker, s=80, alpha=1.0, label=label, color=color, edgecolors='k', linewidths=0.5)
                # -----------------------------------
                
                for i, trade in enumerate(trades_to_plot, start=1):
                    try:
                        # İşlem için zaman aralığını ve olayları belirle
                        entry_dt = trade['entry_dt']
                        exit_dt = trade['exit_dt']
                        entry_px = trade['entry_px']
                        exit_px = trade['exit_px']
                        pnl = trade['pnl']
                        is_long = trade['is_long'] # Stratejiden 'trade.long' olarak geldi
                        
                        # Grafiğe 20 bar (5m*20 = 100dk) öncesi/sonrası boşluk (padding) ekle
                        padding = datetime.timedelta(minutes=(5 * 20))
                        plot_start = entry_dt - padding
                        plot_end = exit_dt + padding
                        
                        # O aralıktaki fiyat verisini al
                        trade_price_data = df_trade.loc[plot_start:plot_end]
                        
                        # Grafiği oluştur
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(trade_price_data.index, trade_price_data['close'], color='gray', alpha=0.7, label='Price (5m)')
                        
                        # Giriş (Entry) işaretini çiz (v33 Düzeltmesi)
                        if is_long:
                            scatter(entry_dt, entry_px, f'Long Entry (Pnl: {pnl:,.2f})', 'lime', ax, marker='^')
                            scatter(exit_dt,  exit_px,  'Long Exit',                         'magenta', ax, marker='x')
                        else: # Short
                            scatter(entry_dt, entry_px, f'Short Entry (Pnl: {pnl:,.2f})', 'red', ax, marker='v')
                            scatter(exit_dt,  exit_px,  'Short Exit',                        'magenta', ax, marker='x')
                        
                        ax.set_title(f'Detaylı İşlem #{i} (Giriş: {entry_dt.isoformat()})')
                        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f"report_trade_{i}.png", dpi=100) # Daha hızlı kaydet
                        plt.close(fig)
                        
                    except Exception as e_trade:
                        print(f"  [Grafik Hatası] İşlem #{i} çizilemedi: {e_trade}")
                        if 'fig' in locals(): plt.close(fig) # Hata olursa figürü kapat
                
                print(f"'Parça Parça' grafikler (report_trade_*.png) kaydedildi.")
        
        except Exception as e: 
            print(f"  [Detaylı Grafik Hatası] 'Parça Parça' grafikler oluşturulamadı: {e}")
            traceback.print_exc()
        # === PARÇA PARÇA GRAFİK BİTİŞ ===


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtrader Strateji Çalıştırıcı (v34 - Meta/OTR)')
    parser.add_argument('--optimize', action='store_true', help='Optimizasyon modunu etkinleştir (Meta-Labeler olmadan)')
    args = parser.parse_args()
    try: run_backtest(optimize=args.optimize)
    except Exception as e: print(f"\n!!! Ana program hatası: {e}"); traceback.print_exc()