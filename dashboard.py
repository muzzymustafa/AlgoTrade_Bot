# dashboard.py — Streamlit Web Dashboard
"""
Kullanım: streamlit run dashboard.py

Gerekli: pip install streamlit plotly
"""
import os
import sys
import glob
import json

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Dashboard için gerekli paketler:")
    print("  pip install streamlit plotly")
    sys.exit(1)


# -----------------------------------------------------------------
# Sayfa ayarları
# -----------------------------------------------------------------
st.set_page_config(
    page_title="AlgoTrade Bot Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AlgoTrade Bot Dashboard")

# -----------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------
st.sidebar.header("Veri Kaynağı")

# Mevcut dosyaları tara
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv")) if os.path.isdir(RESULTS_DIR) else []
trade_files = glob.glob("trades.csv") + glob.glob(os.path.join(RESULTS_DIR, "*trades*.csv"))
state_files = glob.glob(os.path.join(DATA_DIR, "state_*.json")) if os.path.isdir(DATA_DIR) else []
wf_files = glob.glob(os.path.join(RESULTS_DIR, "walk_forward*.csv"))
multi_files = glob.glob(os.path.join(RESULTS_DIR, "multi_symbol*.csv"))


# -----------------------------------------------------------------
# Tab 1: Trade Geçmişi
# -----------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Trade Geçmişi",
    "📈 Equity Curve",
    "🔄 Walk-Forward",
    "🌐 Multi-Symbol",
    "🤖 Paper Trading Durumu",
])

with tab1:
    st.header("Trade Geçmişi")

    # trades.csv veya trades.xlsx
    trades_path = None
    if os.path.exists("trades.csv"):
        trades_path = "trades.csv"
    elif os.path.exists(os.path.join(RESULTS_DIR, "trades.csv")):
        trades_path = os.path.join(RESULTS_DIR, "trades.csv")

    if trades_path:
        df_trades = pd.read_csv(trades_path)
        st.dataframe(df_trades, use_container_width=True)

        if "pnl" in df_trades.columns:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_pnl = df_trades["pnl"].sum()
                st.metric("Toplam PnL", f"${total_pnl:.2f}",
                          delta=f"{'+'if total_pnl>0 else ''}{total_pnl:.2f}")
            with col2:
                win_rate = (df_trades["pnl"] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Toplam Trade", len(df_trades))
            with col4:
                if "exit_reason" in df_trades.columns:
                    most_common = df_trades["exit_reason"].mode().iloc[0] if len(df_trades) else "-"
                    st.metric("En Sık Çıkış", most_common)

            # PnL dağılımı
            fig_pnl = px.histogram(df_trades, x="pnl", nbins=30, title="PnL Dağılımı")
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Kümülatif PnL
            df_trades["cumulative_pnl"] = df_trades["pnl"].cumsum()
            fig_cum = px.line(df_trades, y="cumulative_pnl", title="Kümülatif PnL")
            st.plotly_chart(fig_cum, use_container_width=True)

            # Rejim bazlı analiz
            if "regime" in df_trades.columns:
                st.subheader("Rejim Bazlı Performans")
                regime_stats = df_trades.groupby("regime").agg(
                    trades=("pnl", "count"),
                    total_pnl=("pnl", "sum"),
                    avg_pnl=("pnl", "mean"),
                    win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                ).round(2)
                st.dataframe(regime_stats, use_container_width=True)
    else:
        st.info("Trade verisi bulunamadı. Önce bir backtest çalıştırın.")


with tab2:
    st.header("Equity Curve")

    # Backtest sonuç dosyalarını listele
    opt_files = [f for f in csv_files if "opt" in os.path.basename(f).lower()]

    if opt_files:
        selected = st.selectbox("Optimizasyon Dosyası", opt_files,
                                format_func=lambda x: os.path.basename(x))
        df_opt = pd.read_csv(selected)
        st.dataframe(df_opt.sort_values("net_profit", ascending=False).head(20),
                      use_container_width=True)

        if "net_profit" in df_opt.columns and "sharpe" in df_opt.columns:
            fig = px.scatter(
                df_opt, x="sharpe", y="net_profit",
                color="winrate" if "winrate" in df_opt.columns else None,
                hover_data=df_opt.columns.tolist(),
                title="Sharpe vs Net Profit",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Optimizasyon sonucu bulunamadı. `python main.py --mode baseline --optimize` çalıştırın.")


with tab3:
    st.header("Walk-Forward Sonuçları")

    if wf_files:
        selected_wf = st.selectbox("Walk-Forward Dosyası", wf_files,
                                    format_func=lambda x: os.path.basename(x))
        df_wf = pd.read_csv(selected_wf)
        st.dataframe(df_wf, use_container_width=True)

        if "net_pl" in df_wf.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ort. OOS P/L", f"${df_wf['net_pl'].mean():.2f}")
            with col2:
                profitable = (df_wf["net_pl"] > 0).sum()
                st.metric("Kârlı Pencere", f"{profitable}/{len(df_wf)}")
            with col3:
                if "sharpe" in df_wf.columns:
                    st.metric("Ort. Sharpe", f"{df_wf['sharpe'].fillna(0).mean():.3f}")

            fig_wf = go.Figure()
            fig_wf.add_trace(go.Bar(
                x=df_wf["window"] if "window" in df_wf.columns else df_wf.index,
                y=df_wf["net_pl"],
                marker_color=["green" if x > 0 else "red" for x in df_wf["net_pl"]],
            ))
            fig_wf.update_layout(title="Pencere Bazlı OOS P/L", xaxis_title="Pencere", yaxis_title="Net P/L ($)")
            st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("Walk-forward sonucu yok. `python main.py --walk-forward` çalıştırın.")


with tab4:
    st.header("Multi-Symbol Karşılaştırma")

    if multi_files:
        selected_ms = st.selectbox("Multi-Symbol Dosyası", multi_files,
                                    format_func=lambda x: os.path.basename(x))
        df_ms = pd.read_csv(selected_ms)
        st.dataframe(df_ms, use_container_width=True)
    else:
        st.info("Multi-symbol sonucu yok. `python main.py --multi --mode baseline` çalıştırın.")


with tab5:
    st.header("Paper Trading Durumu")

    if state_files:
        for sf in state_files:
            name = os.path.basename(sf)
            with st.expander(name, expanded=True):
                try:
                    with open(sf, "r") as f:
                        state = json.load(f)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bakiye", f"${state.get('cash', 0):.2f}")
                    with col2:
                        st.metric("Trade Sayısı", state.get("trade_count", 0))
                    with col3:
                        pos = state.get("position")
                        if pos:
                            st.metric("Pozisyon", f"{pos.get('side','?').upper()} {pos.get('size',0):.6f}")
                        else:
                            st.metric("Pozisyon", "YOK")
                    st.json(state)
                except Exception as e:
                    st.error(f"State okuma hatası: {e}")
    else:
        st.info("Paper trading state dosyası yok. `python main.py --paper` çalıştırarak oluşturun.")


# -----------------------------------------------------------------
# Footer
# -----------------------------------------------------------------
st.divider()
st.caption("AlgoTrade Bot Dashboard | Streamlit")
