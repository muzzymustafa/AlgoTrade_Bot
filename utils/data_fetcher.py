# === utils/data_fetcher.py ===
import ccxt
import pandas as pd
import time
import datetime

def fetch_binance_data(symbol, timeframe, total_bars, bars_per_request):
    """
    Binance'ten sayfalama (pagination) yaparak çoklu veri çeker ve Backtrader için hazırlar.
    """
    print(f'Borsa objesi oluşturuluyor: binance')
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # ccxt'nin anlayacağı milisaniye cinsinden zaman aralığı
    timeframe_duration_in_ms = exchange.parse_timeframe(timeframe) * 1000
    all_ohlcv = []
    
    print(f'binance borsasından {symbol} için {timeframe} verisi çekiliyor...')
    print(f'Toplam {total_bars} bar veri {bars_per_request} bar_per_request limit ile çekilecek.')

    try:
        # Ne kadar geriden başlayacağımızı hesapla
        since = exchange.milliseconds() - total_bars * timeframe_duration_in_ms

        while len(all_ohlcv) < total_bars:
            print(f"Çekilen bar sayısı: {len(all_ohlcv)} / {total_bars}")
            
            # Kalan barları veya limiti çek
            limit = min(total_bars - len(all_ohlcv), bars_per_request)
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            # Veri gelmediyse döngüden çık (örn: çok eskiye gidildi)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Bir sonraki istek için 'since' parametresini son barın zamanından başlat
            since = ohlcv[-1][0] + timeframe_duration_in_ms
            
            # Rate limit'e takılmamak için küçük bir bekleme
            time.sleep(exchange.rateLimit / 1000)

        print(f"Veri başarıyla çekildi, toplam {len(all_ohlcv)} bar alındı.")
        
        # Pandas DataFrame'e dönüştür
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Duplike kayıtları (borsa bazen son barı tekrar verebilir) kaldır
        df.drop_duplicates(subset='timestamp', inplace=True)
        
        # === BACKTRADER İÇİN KRİTİK DÖNÜŞÜM ===
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Tam olarak istediğimiz sayıda bar döndürdüğümüzden emin olalım
        final_df = df.tail(total_bars)
        print(f"DataFrame oluşturuldu. Satır sayısı: {len(final_df)}")
        return final_df

    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None