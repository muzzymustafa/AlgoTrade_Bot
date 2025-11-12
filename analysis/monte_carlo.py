# === analysis/monte_carlo.py ===
import numpy as np

def run_monte_carlo(pnl_list, n_simulations=10000):
    """
    Verilen Kâr/Zarar (PnL) listesi üzerinde bir Monte Carlo simülasyonu çalıştırır.
    Bu, stratejinin 'sağlamlığını' (robustness) test eder.
    
    Yöntem: Bootstrapping (Yeniden Örnekleme)
    """
    
    # Kapanan işlem yoksa analizi atla
    if not pnl_list:
        print("\nMonte Carlo: Analiz için yeterli kapanan işlem yok.")
        return

    print("\n" + "="*30)
    print("   MONTE CARLO SAĞLAMLIK ANALİZİ")
    print("="*30)
    print(f"{len(pnl_list)} adet kapanan işlem üzerinden {n_simulations} simülasyon başlatılıyor...")

    # Simülasyon için verileri hazırla
    pnl_array = np.array(pnl_list)
    n_trades_per_sim = len(pnl_list) # Her simülasyon, gerçekteki işlem sayısı kadar olsun
    
    simulated_results = [] # Tüm simülasyonların net kâr/zarar sonuçlarını tutar

    # Simülasyon döngüsü
    for i in range(n_simulations):
        # Orijinal PnL listesinden rastgele (yerine koyarak) işlem seç
        sim_trades = np.random.choice(pnl_array, n_trades_per_sim, replace=True)
        
        # Bu rastgele seçilmiş işlemlerin toplam kârını/zararını hesapla
        sim_pnl = np.sum(sim_trades)
        simulated_results.append(sim_pnl)

    # 4. Sonuçları Analiz Et
    mean_pnl = np.mean(simulated_results)
    std_dev = np.std(simulated_results)
    
    # Yüzdelik dilimler (Güven Aralığı)
    p5_percentile = np.percentile(simulated_results, 5)  # En kötü %5'lik senaryo (VaR benzeri)
    p25_percentile = np.percentile(simulated_results, 25)
    p75_percentile = np.percentile(simulated_results, 75)
    p95_percentile = np.percentile(simulated_results, 95) # En iyi %5'lik senaryo
    
    # Zarar etme olasılığı
    prob_loss = np.mean(np.array(simulated_results) < 0) * 100

    # 5. Raporu Yazdır
    print(f"\nOrijinal Backtest Net Kârı: {np.sum(pnl_array):.2f}")
    print(f"Simülasyon Ortalaması (Mean): {mean_pnl:.2f}")
    print(f"Standart Sapma: {std_dev:.2f}")
    
    print("\nGüven Aralığı (Yüzdelik Dilimler):")
    print(f" %5'lik Dilim (En Kötü): {p5_percentile:.2f}")
    print(f" %25'lik Dilim: {p25_percentile:.2f}")
    print(f" %75'lik Dilim: {p75_percentile:.2f}")
    print(f" %95'lik Dilim (En İyi): {p95_percentile:.2f}")
    
    print("\nRisk Analizi:")
    print(f" Zarar Etme Olasılığı: {prob_loss:.2f}%")
    print("="*30)
    
    if p5_percentile > 0:
        print("SONUÇ: Strateji SAĞLAM görünüyor. Simülasyonların %95'i kâr ile sonuçlandı.")
    elif mean_pnl > 0 and prob_loss < 50:
        print("SONUÇ: Strateji kârlı olma eğiliminde ancak risk içeriyor.")
    else:
        print("SONUÇ: Strateji YÜKSEK RİSKLİ veya şansa bağlı görünüyor. Dikkatli olun.")
    print("="*30 + "\n")