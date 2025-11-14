# features/build_features.py (v32)
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# ------------------------------------------------------------
# Yardımcı fonksiyonlar (Wilder ADX, ATR, ROC, rolling z-score)
# ------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    # Welles Wilder RMA (Smoothed Moving Average)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def roc(close: pd.Series, period: int = 20) -> pd.Series:
    return close.pct_change(periods=period)

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # +DM / -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = _true_range(high, low, close)

    # Wilder smoothing (RMA)
    tr_rma = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_rma = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_rma = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm_rma / tr_rma).replace({0: np.nan})
    minus_di = 100 * (minus_dm_rma / tr_rma).replace({0: np.nan})

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace({0: np.nan}) ) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    """
    Gelecekten bilgi sızdırmamak için scaler yerine rolling mean/std.
    """
    mean = s.rolling(win, min_periods=win//2).mean()
    std = s.rolling(win, min_periods=win//2).std(ddof=0)
    return (s - mean) / std

# ------------------------------------------------------------
# Donchian Kanalı
# ------------------------------------------------------------

def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    hi = df['high'].rolling(period, min_periods=period).max()
    lo = df['low'].rolling(period, min_periods=period).min()
    mid = (hi + lo) / 2.0
    width = (hi - lo) / df['close']  # normalize genişlik
    out = pd.DataFrame({
        'donchian_upper': hi,
        'donchian_lower': lo,
        'donchian_mid': mid,
        'donchian_width': width
    }, index=df.index)
    return out

# ------------------------------------------------------------
# SMA gücü (basit çapraz kuvveti)
# ------------------------------------------------------------

def cross_strength(close: pd.Series, fast: int, slow: int) -> pd.Series:
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    # normalize: fark / slow SMA
    strength = (sma_fast - sma_slow) / sma_slow
    return strength

# ------------------------------------------------------------
# Ana özellik oluşturucu
# ------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cfg
) -> pd.DataFrame:
    """
    Girdi: OHLCV içeren, datetime index'li DataFrame
      gerekli kolonlar: ['open','high','low','close','volume']

    Çıktı: Özellikler + (opsiyonel) meta_label
    Veri sızıntısı önlemleri:
      - Tüm istatistikler rolling ile hesaplanır.
      - Z-score için rolling window kullanılır (fit yok).
      - meta_label varsa, geleceğe kaydırma (shift(-H)) yapılır ve
        son H bar düşürülür.
    """
    df = df.copy()

    # --- Parametreleri al ---
    atr_p = getattr(cfg, 'ATR_PERIOD', 14)
    roc_p = getattr(cfg, 'KMEANS_ROC_PERIOD', 20)
    adx_p = getattr(cfg, 'ADX_PERIOD', 14)
    don_p = getattr(cfg, 'DONCHIAN_PERIOD', 20)
    zwin  = getattr(cfg, 'FEAT_ZWIN', 500)

    tf_fast = getattr(cfg, 'TRADE_FAST_SMA', 10)
    tf_slow = getattr(cfg, 'TRADE_SLOW_SMA', 50)

    # --- Temel indikatörler ---
    df['atr'] = atr_wilder(df['high'], df['low'], df['close'], period=atr_p)
    df['roc'] = roc(df['close'], period=roc_p)

    # --- ADX ---
    df['adx'] = adx_wilder(df['high'], df['low'], df['close'], period=adx_p)

    # --- Donchian ---
    dch = donchian(df, period=don_p)
    df = df.join(dch)

    # Donchian width için da rolling z-score (ölçekleme)
    df['donchian_width_z'] = rolling_zscore(df['donchian_width'], zwin)

    # --- K-Means özellikleri için ölçekli versiyonlar (rolling z) ---
    df['ATR_scaled'] = rolling_zscore(df['atr'], zwin)
    df['ROC_scaled'] = rolling_zscore(df['roc'], zwin)

    # --- SMA cross kuvveti (LTF) ---
    df['cross_strength'] = cross_strength(df['close'], tf_fast, tf_slow)

    # --- Filtrelere yardımcı alanlar ---
    # ADX filtre: cfg.ADX_MIN değerlendirmesini strategy tarafında yap
    df['adx_ok'] = df['adx'] >= getattr(cfg, 'ADX_MIN', 20)

    # Rejim seçimi için (strategy/engine kullanır)
    df['regime_feature1'] = df['ATR_scaled']
    df['regime_feature2'] = df['ROC_scaled']

    # ------------------------------------------------------------
    # Opsiyonel: Basit meta_label
    # Gelecek H bar getirisi > 0 ise 1, değilse 0
    # (triple-barrier yerine temel etiketleme; sızıntı yok)
    # ------------------------------------------------------------
    use_meta = getattr(cfg, 'USE_META_LABELER', False)
    if use_meta:
        H = getattr(cfg, 'META_HORIZON_BARS', 12)
        fwd_ret = df['close'].shift(-H) / df['close'] - 1.0
        df['meta_label'] = (fwd_ret > 0).astype('int')
        # Son H bar geleceğe bakar — eğitime dahil edilmemeli:
        df.iloc[-H:, df.columns.get_loc('meta_label')] = np.nan

    # Başlarda rolling kaynaklı NaN'ları temizle
    # (Donchian ve SMA için yeterli doluluk bekleriz)
    min_req = max(atr_p, roc_p, adx_p, don_p, tf_fast, tf_slow, zwin//2)
    df = df.iloc[min_req:].copy()

    # Eğer meta_label varsa NaN’ları at:
    if use_meta:
        df = df.dropna(subset=['meta_label'])

    return df

# ------------------------------------------------------------
# K-Means matrisini çıkar (opsiyonel yardımcı)
# ------------------------------------------------------------

def make_kmeans_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means'e verilecek sütunları döndürür.
    Burada zaten rolling z ile ölçekli sütunlar var.
    """
    cols = ['ATR_scaled', 'ROC_scaled']
    return feat_df[cols].dropna()
