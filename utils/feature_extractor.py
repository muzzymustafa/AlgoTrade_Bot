# === GÜNCELLENDİ: utils/feature_extractor.py ===
import pandas_ta as ta
import numpy as np
import pandas as pd
import zlib # Lempel-Ziv için

def add_kmeans_features(df, atr_period, roc_period):
    """
    DataFrame'e K-Means modeli için özellikler (features) ekler.
    Kullanılan özellikler:
    1. ATR (Average True Range) - Volatilite ölçer
    2. ROC (Rate of Change) - Momentum ölçer
    """
    print("K-Means özellikleri (ATR, ROC) hesaplanıyor...")
    
    # pandas-ta kütüphanesi ile özellikleri hesapla
    df.ta.atr(length=atr_period, append=True)
    df.ta.roc(length=roc_period, append=True)
    
    # Özellik adlarını daha okunabilir yapalım (örn: ATR_14, ROC_20)
    atr_col = f'ATRr_{atr_period}'
    roc_col = f'ROC_{roc_period}'
    df.rename(columns={atr_col: 'ATR', roc_col: 'ROC'}, inplace=True)

    # Bu indikatörler ilk başta NaN (boş) değerler üretir.
    # Bu boş satırları temizlemezsek model hata verir.
    df.dropna(inplace=True)
    
    # Modeli eğitmek için sadece özellik sütunlarını ayrıca döndür
    features = df[['ATR', 'ROC']]
    
    print(f"Özellikler eklendi, NaN satırlar temizlendi. Kalan veri: {len(df)}")
    
    return df, features

# --- YENİ: FracDiff ve Zengin Özellikler ---

def fracdiff(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Kesirli Fark Alma (Fractional Differentiation)
    López de Prado'nun 'sabit pencere' (expanding window değil) yöntemini uygular.
    Seriyi durağanlaştırırken hafızayı (memory) korur.
    
    d: Fark alma derecesi (0 ile 1 arası)
    thres: Ağırlıklar (weights) için kesme eşiği
    """
    # 1. Ağırlıkları (weights) hesapla
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])  # En eskiden -> en yeniye sırala
    n_weights = len(w)
    
    # 2. Seriye uygula
    out = pd.Series(index=series.index, dtype='float64')
    
    # Numpy dot product (vektörel çarpım) kullanarak hızlı hesaplama
    for i in range(n_weights, len(series) + 1):
        # O anki pencereyi al
        window = series.iloc[i - n_weights : i].values
        # Ağırlıklarla çarp
        out.iloc[i - 1] = np.dot(w, window)
        
    return out

def lz_complexity(x: pd.Series, bins: int = 8) -> pd.Series:
    """
    Lempel-Ziv Karmaşıklığının kaba bir tahmini.
    Serinin ne kadar 'rastgele' veya 'öngörülebilir' olduğunu ölçer.
    Düşük karmaşıklık = trend/momentum (daha sıkıştırılabilir)
    Yüksek karmaşıklık = gürültü/chop (daha az sıkıştırılabilir)
    """
    # Seriyi ayrıklaştır (quantize)
    q = pd.qcut(x.dropna(), bins, labels=False, duplicates='drop')
    q = q.reindex(x.index).fillna(method='ffill').astype(int)
    
    # String'e çevir
    comp = q.astype(str).str.cat(sep='')
    
    # zlib ile sıkıştır ve sıkıştırma oranını hesapla
    c = len(zlib.compress(comp.encode('utf-8')))
    
    # Orijinal serinin uzunluğunda bir Seri olarak döndür
    return pd.Series([c / len(comp)] * len(x), index=x.index)

def build_features(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Meta-Labeler için 5m zaman serisi üzerinde zengin özellik matrisi oluşturur.
    """
    print("[BuildFeatures] Meta-model için özellikler oluşturuluyor...")
    
    out = pd.DataFrame(index=df_5m.index)
    
    # 1. Getiriler ve Mikroyapı
    r = df_5m['close'].pct_change()
    out['ret1']  = r
    out['ret5']  = df_5m['close'].pct_change(5)
    out['ret10'] = df_5m['close'].pct_change(10)
    
    # 2. Volatilite / Oynaklık
    # 30 barlık (2.5 saat) gerçekleşen volatilite (yıllıklandırılmış)
    out['rv_30'] = r.rolling(30).std() * np.sqrt(288 * 252) # 288 bar/gün
    out['atr_14'] = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
    
    # 3. Momentum
    out['roc_60'] = df_5m['close'].pct_change(60) # 5 saatlik değişim
    
    # 4. Hafıza ve Karmaşıklık (Gelişmiş Özellikler)
    # Fiyat serisi (log) üzerinde FracDiff
    out['fd_d05'] = fracdiff(np.log(df_5m['close']), d=0.5)
    # Getiri serisi üzerinde LZ Karmaşıklığı
    out['lz']     = lz_complexity(r.fillna(0))
    
    # 5. Rejim (1h verisinden)
    # 1h'lik veriyi 5m'lik indekse 'ffill' (ileri doldurma) ile hizala
    if 'regime' in df_1h.columns:
        reg_hourly = df_1h[['regime']].reindex(df_5m.index, method='ffill')
        out['regime'] = reg_hourly['regime']
        
    print("[BuildFeatures] Özellikler oluşturuldu, NaN'lar temizleniyor.")
    return out.dropna()