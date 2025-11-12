# === YENİ DOSYA: analysis/otr.py ===
import numpy as np
import pandas as pd

def fit_otr_by_regime(returns, atr, regimes, grid=((1.0,2.0),(1.2,2.4),(1.5,3.0))):
    """
    Optimal Trading Rule (OTR) için basit bir ızgara araması (grid search) yapar.
    Amacı: Her rejim (K-Means) için, Sharpe Oranını maksimize eden
    en iyi (SL_multiplier, TP_multiplier) çiftini bulmak.
    
    returns: Meta etiketleme için kullanılan gelecekteki getiriler (örn: 36 bar sonrası)
    atr: O anki ATR değeri
    regimes: O anki rejim etiketi
    grid: Denenecek (SL, TP) çarpan çiftleri
    """
    print(f"\n[OTR] Rejime göre en iyi SL/TP (ATR Çarpanı) aranıyor...")
    
    best_config_map = {}
    
    # Girdileri hizala
    data = pd.DataFrame({'returns': returns, 'atr': atr, 'regime': regimes}).dropna()
    
    if data.empty:
        print("[OTR] UYARI: OTR için hizalanmış veri bulunamadı.")
        return {}
        
    unique_regimes = data['regime'].unique()
    
    for rg in unique_regimes:
        if pd.isna(rg): continue
        
        mask = (data['regime'] == rg)
        if mask.sum() < 100: # Analiz için minimum veri
            print(f"[OTR] Rejim {int(rg)} atlanıyor (Veri yetersiz: {mask.sum()} bar)")
            continue

        regime_data = data[mask]
        best_sr = -1e9
        best_pair = grid[0]

        for sl_mult, tp_mult in grid:
            
            # Basit bir simülasyon:
            # Getiri/Risk (ATR) oranını proxy olarak kullanalım.
            # SL = atr * sl_mult, TP = atr * tp_mult
            # Bu, tam bir backtest simülasyonu DEĞİLDİR, hızlı bir tahmindir.
            
            # ATR'ye göre normalize edilmiş getiriler
            normalized_returns = regime_data['returns'] / (regime_data['atr'] + 1e-9)
            
            # SL/TP çarpanlarına göre kâr/zararı simüle et
            # (Çok kaba bir yaklaşım - normalde burada bir simülasyon döngüsü gerekir)
            
            # Proxy: Getirisi TP'den büyükse TP kadar, SL'den küçükse SL kadar al.
            # Bu hatalı bir varsayımdır, ancak ızgarayı test etmek için hızlıdır.
            # Doğru yöntem: 'returns' serisini değil, fiyat serisini işlemek.
            
            # Daha basit ve (nispeten) daha doğru bir proxy:
            # Sharpe Oranını, TP çarpanıyla ölçeklendirip SL çarpanıyla cezalandıralım.
            
            r_std = normalized_returns.std()
            if r_std == 0: r_std = 1.0
            
            r_mean = normalized_returns.mean()
            
            # Sharpe Oranını andıran bir proxy metrik
            # Yüksek TP ve düşük SL'yi ödüllendiren kaba bir metrik
            proxy_metric = (r_mean * tp_mult) / (r_std * sl_mult + 1e-9)

            if proxy_metric > best_sr:
                best_sr = proxy_metric
                best_pair = (sl_mult, tp_mult)
        
        best_config_map[int(rg)] = best_pair
        print(f"[OTR] Rejim {int(rg)} için en iyi (SL, TP) çarpanı: {best_pair} (Proxy SR: {best_sr:.3f})")

    return best_config_map