# === YENİ DOSYA: ml/meta_labeler.py ===
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils.validation import PurgedKFold

class MetaLabeler:
    """
    Meta-Etiketleme Modeli.
    
    1. Ana stratejiden (SMA Cross) bir sinyal alır (Long/Short).
    2. Bu model (örn: RandomForest), bu sinyalin kârlı olup olmayacağını
       tahmin eder (ikincil model).
    3. Tahmin olasılığına (proba) göre pozisyon büyüklüğünü (bet size) belirler.
    """
    
    def __init__(self, clf=None, threshold=0.55):
        self.clf = clf or RandomForestClassifier(
            n_estimators=300, 
            max_depth=7,        # Biraz daha derin
            random_state=42, 
            n_jobs=-1,
            min_samples_leaf=10 # Aşırı öğrenmeyi engelle
        )
        self.threshold = threshold
        self.fitted = False
        self.features = None # Eğitimde kullanılan özellik listesi

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Modeli eğitir.
        X: Özellikler (features)
        y: Etiketler (labels) (0 veya 1 - Kâr etti mi?)
        """
        print(f"\n[MetaLabeler] Model {X.shape[0]} örnek üzerinde eğitiliyor...")
        
        # PurgedKFold CV (Eğitim için kullanılabilir, ancak şimdilik direkt fit)
        # cv = PurgedKFold(n_splits=5, embargo=0.01)
        # ... (out-of-fold tahminler burada yapılabilir) ...
        
        # (Şimdilik basitlik için tüm eğitim setine fit ediyoruz)
        self.clf.fit(X.values, y.values)
        self.fitted = True
        self.features = list(X.columns) # Özellik sırasını kaydet
        
        # Özellik önemlerini yazdır
        try:
            imp = pd.Series(self.clf.feature_importances_, index=self.features).sort_values(ascending=False)
            print("[MetaLabeler] Model eğitildi. Özellik Önemleri (Top 5):")
            print(imp.head(5).to_string())
        except: pass
        
        return self

    def proba(self, x_row: pd.Series) -> float:
        """ Tek bir bar için kâr olasılığını tahmin eder. """
        if not self.fitted: 
            return 1.0 # Model yoksa, ana stratejiye güven (girişe izin ver)
        
        try:
            # Eğitimdeki özellik sırasına göre veriyi hazırla
            x_aligned = x_row[self.features].values.reshape(1, -1)
            # 1 (Kârlı) olma olasılığını döndür
            return float(self.clf.predict_proba(x_aligned)[0, 1])
        except Exception as e:
            print(f"[MetaLabeler] Tahmin hatası: {e}. Sıralama: {self.features}, Gelen: {x_row.index}")
            return 0.0 # Hata varsa, sinyali engelle

    def bet_size(self, p: float) -> float:
        """
        López de Prado'nun pozisyon büyüklüğü formülü.
        Olasılığa (p) göre 0 (işlem yok) ile 1 (tam pozisyon) arasında
        bir büyüklük döndürür.
        p=0.5 -> 0 (işlem yok)
        p=1.0 -> 1 (tam pozisyon)
        """
        if p < self.threshold: # Model yeterince emin değilse
             return 0.0 
             
        # Güveni 0-1 arasına ölçeklendir (threshold=0.55 ise p=0.55 -> 0, p=1.0 -> 1)
        scaled_p = (p - self.threshold) / (1.0 - self.threshold)
        
        # Orijinal formül: s = 2*p - 1 (p=0.5'te merkezlenir)
        # Bizimki threshold'da merkezleniyor:
        size = 2 * scaled_p - 1
        
        return max(0.0, min(1.0, size))