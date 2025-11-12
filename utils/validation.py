# === YENİ DOSYA: utils/validation.py ===
import numpy as np
from sklearn.model_selection import KFold

class PurgedKFold:
    """
    López de Prado tarafından önerilen, 'Purging' (Temizleme) ve 
    'Embargo' (Yasaklama) uygulayan K-Fold Çapraz Doğrulama.
    Finansal verilerdeki seri korelasyon ve bilgi sızıntısını önler.
    """
    def __init__(self, n_splits=5, embargo=0.01):
        self.kf = KFold(n_splits=n_splits, shuffle=False)
        # Embargo: Eğitim ve test setleri arasına konulacak boşluk (veri yüzdesi)
        self.embargo = embargo

    def split(self, X, event_times=None):
        """
        event_times: Her bir etiketin (y) ne zaman bittiğini gösteren 
                     Pandas Serisi (indeks X ile aynı olmalı).
                     Şimdilik basit bir "yakınlık" temizlemesi kullanıyoruz.
        """
        n = len(X)
        m = int(n * self.embargo) # Embargo bar sayısı
        
        for train_idx, test_idx in self.kf.split(X):
            test_start, test_end = test_idx[0], test_idx[-1]
            
            # purge overlap (Temizleme)
            train_mask = np.ones(n, dtype=bool)
            
            # 1. Test setine çok yakın olan eğitim verilerini kaldır (Purge)
            # (Basitleştirilmiş versiyon: Test setinin etrafındaki 'm' barı temizle)
            purge_start = test_start - m
            purge_end = test_end + m + 1
            train_mask[max(0, purge_start) : min(n, purge_end)] = False
            
            # 2. Test setini de eğitimden çıkar
            train_mask[test_idx] = False

            # 3. Embargo: Test setinden *sonraki* 'm' barı da eğitimden çıkar
            # (Bu, 'purge_end' ile zaten kapsandı)
            
            yield np.where(train_mask)[0], test_idx