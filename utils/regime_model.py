# === utils/regime_model.py ===
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_regime_labels(features, n_clusters):
    """
    Özellikleri alır, veriyi ölçeklendirir ve bir K-Means modelini eğitir.
    Her satır için bir rejim etiketi (0, 1, 2...) döndürür.
    """
    print("K-Means modeli eğitiliyor...")
    
    # 1. Veriyi Ölçeklendir (Çok önemli)
    # ATR (örn: 500) ve ROC (örn: 0.05) farklı ölçeklerdedir.
    # StandardScaler, ikisini de ortalaması 0, std sapması 1 olan bir forma sokar.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. K-Means Modelini Eğit
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(features_scaled)
    
    # 3. Analiz için eğitilmiş modeli ve ölçekleyiciyi döndür
    labels = model.labels_
    print("K-Means modeli eğitildi.")
    
    return model, scaler, labels, features_scaled

def analyze_regimes(model, scaler, features_scaled, labels):
    """
    Eğitilen modelin bulduğu rejimlerin ne anlama geldiğini konsola yazdırır.
    Bu, 'GOOD_REGIME_ID'yi belirlememiz için kritik öneme sahiptir.
    """
    print("\n" + "="*30)
    print("      K-MEANS REJİM ANALİZİ")
    print("="*30)
    
    # Ölçeklenmiş veriye etiketleri ekle
    analysis_df = pd.DataFrame(features_scaled, columns=['ATR_scaled', 'ROC_scaled'])
    analysis_df['Regime'] = labels
    
    # Her rejimin ortalama ATR ve ROC değerini (merkezlerini) hesapla
    regime_centers = analysis_df.groupby('Regime').mean()
    
    print("Bulunan Rejimlerin Özellikleri (Ortalama Değerler):\n")
    print(regime_centers)
    print("\n--- YORUMLAMA ---")
    print("Stratejimiz 'Trend Takip' stratejisidir (SMA Cross).")
    print("Trendin olduğu, yani *Yüksek Momentum (ROC)* olan rejimleri arıyoruz.")
    print("Volatilite (ATR) ne çok düşük (piyasa ölü) ne de çok yüksek (panik) olmalı.")
    print("\nLütfen yukarıdaki tabloya bakın:")
    print(" - 'Yüksek ROC_scaled' değerine sahip rejimi bulun.")
    print(" - Bu rejimin numarasını (0, 1, veya 2) not alın.")
    print(" - 'config.py' dosyasını açıp 'GOOD_REGIME_ID' değişkenini bu numara ile güncelleyin.")
    print("="*30 + "\n")