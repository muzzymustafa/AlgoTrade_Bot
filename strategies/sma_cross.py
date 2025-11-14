# strategy/sma_cross.py (v32)
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd

def cross_strength(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """SMA farkını normalleştirerek cross gücü (~z) tahmini."""
    diff = fast - slow
    roll = diff.rolling(200, min_periods=50)
    z = (diff - roll.mean()) / (roll.std(ddof=0) + 1e-9)
    return z.clip(-5, 5)

def allow_entry(row: pd.Series, cfg) -> bool:
    """Filtreler: ADX, Donchian genişliği, rejim ve min cross gücü."""
    # ADX
    adx_min = getattr(cfg, 'ADX_MIN', None)
    if adx_min is not None:
        adx_ok = row.get('adx_ok', False)
        if not bool(adx_ok):
            return False

    # Donchian genişliği (ölü/panik filtreleri)
    # build_features içinde donchian_width_z üretildi.
    wd = row.get('donchian_width_z', np.nan)
    if not np.isnan(wd):
        # çok-ölü piyasa: wd < -0.5 ; aşırı panik: wd > 2.5 (başlangıç için)
        if wd < -0.5 or wd > 2.5:
            return False

    # Rejim filtresi
    g = getattr(cfg, 'GOOD_REGIME_ID', None)
    r = row.get('regime', np.nan)
    if g is not None and not np.isnan(r):
        if int(r) != int(g):
            return False

    # Cross gücü
    if row.get('cross_strength', 0.0) < getattr(cfg, 'MIN_CROSS_STRENGTH', 0.25):
        return False

    return True

def get_sl_tp_atr(row: pd.Series, cfg):
    """Rejime özel (sl,tp) ATR çarpanı, yoksa statik yüzdelik."""
    atr = row.get('ATR', np.nan)
    price = row.get('close', np.nan)
    if np.isnan(atr) or np.isnan(price):
        # fallback: statik
        sl = price * (1 - getattr(cfg, 'STOP_LOSS', 0.02))
        tp = price * (1 + getattr(cfg, 'TAKE_PROFIT', 0.05))
        return sl, tp

    reg = row.get('regime', np.nan)
    lut = getattr(cfg, 'REGIME_SL_TP', {})
    mult = lut.get(int(reg) if not np.isnan(reg) else None, None)

    if mult is None:
        # fallback statik
        sl = price * (1 - getattr(cfg, 'STOP_LOSS', 0.02))
        tp = price * (1 + getattr(cfg, 'TAKE_PROFIT', 0.05))
        return sl, tp

    sl_mult, tp_mult = mult
    sl = price - sl_mult * atr
    tp = price + tp_mult * atr
    return sl, tp

def generate_signals(feat: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Girdi: feat (build_features çıktısı + 'regime')
    Çıktı: signals df (entry/exit + hedefler)
    """
    df = feat.copy()

    # Trend penceresi: SMA(fast/slow) zaten build_features'ta var varsayıyoruz:
    # 'sma_fast_trade', 'sma_slow_trade', 'sma_fast_trend', 'sma_slow_trend'
    for col in ['sma_fast_trade','sma_slow_trade','sma_fast_trend','sma_slow_trend']:
        if col not in df.columns:
            raise ValueError(f"features eksik: {col}")

    # Trade cross & strength
    df['trade_cross_up'] = (df['sma_fast_trade'] > df['sma_slow_trade']) & (df['sma_fast_trade'].shift(1) <= df['sma_slow_trade'].shift(1))
    df['trade_cross_dn'] = (df['sma_fast_trade'] < df['sma_slow_trade']) & (df['sma_fast_trade'].shift(1) >= df['sma_slow_trade'].shift(1))
    df['cross_strength'] = cross_strength(df['sma_fast_trade'], df['sma_slow_trade'])

    # Trend filtresi: sadece up-trend’de long
    df['in_trend_up'] = df['sma_fast_trend'] > df['sma_slow_trend']

    # Cooldown
    cool = int(getattr(cfg, 'COOL_DOWN_BARS', 12))
    df['cooldown'] = False
    last_entry_idx = None

    entries = []
    exits = []
    sls = []
    tps = []

    position = 0  # 0: flat, 1: long
    since_entry = 0

    for i, row in df.iterrows():
        # cooldown penceresini güncelle
        if last_entry_idx is not None:
            since_entry = (df.index.get_loc(i) - df.index.get_loc(last_entry_idx))
            if since_entry < cool:
                df.at[i, 'cooldown'] = True

        if position == 0:
            # sadece long taraf
            if row['in_trend_up'] and row['trade_cross_up'] and not df.at[i, 'cooldown']:
                if allow_entry(row, cfg):
                    # entry
                    position = 1
                    last_entry_idx = i
                    sl, tp = get_sl_tp_atr(row, cfg)
                    entries.append(i)
                    sls.append(sl); tps.append(tp)
                    exits.append(pd.NaT)
                else:
                    entries.append(pd.NaT); exits.append(pd.NaT); sls.append(np.nan); tps.append(np.nan)
            else:
                entries.append(pd.NaT); exits.append(pd.NaT); sls.append(np.nan); tps.append(np.nan)

        else:  # position == 1
            # SL/TP kontrolü (bar içi yaklaşıklık: close bazlı)
            price = row['close']
            sl = sls[-1]; tp = tps[-1]

            exit_now = False
            if not np.isnan(sl) and price <= sl:
                exit_now = True
            if not np.isnan(tp) and price >= tp:
                exit_now = True

            # ters kesişme (trade_cross_dn) de çıkış sebebi
            if row['trade_cross_dn']:
                exit_now = True

            if exit_now:
                position = 0
                exits.append(i)
                # SL/TP reset
                sls.append(np.nan); tps.append(np.nan)
                entries.append(pd.NaT)
            else:
                entries.append(pd.NaT); exits.append(pd.NaT); sls.append(sl); tps.append(tp)

    out = pd.DataFrame(index=df.index)
    out['entry'] = entries
    out['exit'] = exits
    out['sl'] = sls
    out['tp'] = tps
    return out
