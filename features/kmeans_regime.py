# features/kmeans_regime.py (v32)
# -*- coding: utf-8 -*-
# DEPRECATED: Bu modül eski bir versiyondur.
# Canonical versiyon: utils/regime_model.py (get_regime_labels, analyze_regimes)
# Bu dosya geriye uyumluluk için tutulmaktadır.

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    # sklearn opsiyonel olabilir; yoksa ImportError yakalanır
    from sklearn.cluster import KMeans
except Exception as e:
    KMeans = None

def get_kmeans_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    """K-Means’e girecek sütunları hazırla."""
    cols = ['ATR_scaled', 'ROC_scaled']
    X = feat_df[cols].dropna()
    return X

def fit_kmeans(feat_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    if KMeans is None:
        raise ImportError("scikit-learn gerekli: pip install scikit-learn")

    X = get_kmeans_matrix(feat_df)
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
    labels = pd.Series(km.fit_predict(X.values), index=X.index, name='regime')
    return km, labels

def regime_stats(feat_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Her rejimin ATR/ROC ortalamalarını ver."""
    df = feat_df.loc[labels.index].copy()
    df['regime'] = labels
    g = df.groupby('regime')[['ATR_scaled','ROC_scaled']].mean()
    # okunaklılık için sırala: ROC desc
    g = g.sort_values('ROC_scaled', ascending=False)
    return g

def auto_select_good_regime(stats_df: pd.DataFrame) -> int:
    """En yüksek ROC ortalamasına sahip rejimi seç."""
    return int(stats_df.index[0])

def label_full_index(full_index: pd.Index, labels: pd.Series) -> pd.Series:
    """Label’ları tüm feature index’ine reindex et (eksikler NaN)."""
    lab = labels.reindex(full_index)
    return lab

def run_kmeans_regime(feat_df: pd.DataFrame, cfg):
    """
    Çıktılar:
      - labels_full: feat_df.index ile hizalı rejim etiketleri (NaN olabilir)
      - stats_df   : rejim özet tablosu (ATR/ROC ort.)
      - good_id    : seçilen iyi rejim
    """
    n_clusters = getattr(cfg, 'KMEANS_N_CLUSTERS', 3)
    km, labels = fit_kmeans(feat_df, n_clusters=n_clusters)
    stats_df = regime_stats(feat_df, labels)

    # İyi rejim seçimi
    if getattr(cfg, 'KMEANS_AUTO_SELECT', True):
        good_id = auto_select_good_regime(stats_df)
    else:
        good_id = getattr(cfg, 'GOOD_REGIME_ID', None)
        if good_id is None:
            good_id = auto_select_good_regime(stats_df)

    labels_full = label_full_index(feat_df.index, labels)
    return labels_full, stats_df, good_id
