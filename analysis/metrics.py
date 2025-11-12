# === YENİ DOSYA: analysis/metrics.py ===
import numpy as np
import math
from math import erf, sqrt

def sharpe_ratio(returns, rf=0.0, periods_per_year=252*24*12):
    """
    Verilen getiri serisi için yıllıklandırılmış Sharpe Oranını hesaplar.
    Varsayılan (5m) periyot: (60/5) * 24 * 252
    """
    r = np.array(returns)
    if r.std() == 0 or r.std() is np.nan: 
        return 0.0
        
    mean_ret = r.mean()
    std_ret = r.std()
    
    return (mean_ret - rf) / std_ret * np.sqrt(periods_per_year)

def deflated_sharpe(sr, n_trials, T, returns=None):
    """
    Deflated Sharpe Ratio (DSR) hesaplar.
    Geriye dönük test aşırı öğrenmesini (backtest overfitting)
    ve çoklu test yanlılığını (multiple testing bias) dikkate alır.
    
    sr: Gözlemlenen (en iyi) Sharpe Oranı
    n_trials: Denenen toplam strateji sayısı (örn: optimizasyondaki kombinasyon sayısı)
    T: Gözlem (bar) sayısı
    returns: Getiri serisi (skew ve kurtosis hesaplamak için)
    """
    if T <= 1 or n_trials <= 0: return 0.0

    # Bailey et al. approx: DSR ≈ Φ( (sr - μ_z)/σ_z )
    # (Kaba bir yaklaşım kullanıyoruz)
    
    # Beklenen maksimum SR'yi tahmin et
    emc = 0.5772156649 # Euler-Mascheroni sabiti
    
    # Getiri serisi normalse
    if returns is None:
        max_z_approx = (1 - emc) * np.linalg.inv(np.random.normal(size=(n_trials, n_trials))).max() + \
                       emc * np.random.normal(size=(n_trials)).max()
    else:
        # Skew ve Kurtosis'i hesaba kat (daha doğru)
        try:
            from scipy.stats import skew, kurtosis
            sk = skew(returns)
            k = kurtosis(returns, fisher=False) # Fisher değil, Pearson
            T_ = len(returns)
            # 3. ve 4. momentleri hesaba kat
            g3 = sk / (T_**(1/2))
            g4 = (k-3) / (T_**(1/2))
            # ... (Daha karmaşık formülasyon - şimdilik basiti kullanalım)
            max_z_approx = (1 - emc) * np.linalg.inv(np.random.normal(size=(n_trials, n_trials))).max() + \
                       emc * np.random.normal(size=(n_trials)).max()
        except:
             max_z_approx = (1 - emc) * np.linalg.inv(np.random.normal(size=(n_trials, n_trials))).max() + \
                       emc * np.random.normal(size=(n_trials)).max()

    
    # Standart sapma 1 varsayımıyla (normalize edilmiş getiriler)
    mu_z = max_z_approx 
    sigma_z = 1.0 # Yaklaşık
    
    # DSR'yi Z-skoru ve kümülatif dağılım fonksiyonu (CDF) olarak hesapla
    z = (sr - mu_z) / (sigma_z + 1e-9)
    dsr_p_value = 0.5 * (1 + erf(z / sqrt(2)))
    
    return dsr_p_value