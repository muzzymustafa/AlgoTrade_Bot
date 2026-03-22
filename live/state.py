# live/state.py
"""Bot durumu — JSON dosyasına kaydedilir, yeniden başlatmada yüklenir."""
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class State:
    # Pozisyon
    position: dict | None = None  # {side, size, entry_price, entry_bar}
    pending_sl_id: str | None = None
    pending_tp_id: str | None = None

    # Zamanlama
    last_bar_time: str | None = None  # ISO format
    bars_since_last_trade: int = 0
    entry_bar: int = 0
    entry_price: float = 0.0

    # Paper broker state
    cash: float = 10000.0
    trade_count: int = 0

    # BE stop
    be_stop_active: bool = False

    def save(self, path: str):
        """Durumu JSON dosyasına kaydet."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "State":
        """JSON dosyasından yükle. Dosya yoksa varsayılan döndür."""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[State] Yükleme hatası, varsayılan kullanılıyor: {e}")
            return cls()

    def reset_position(self):
        """Pozisyon kapandığında state'i temizle."""
        self.position = None
        self.pending_sl_id = None
        self.pending_tp_id = None
        self.entry_bar = 0
        self.entry_price = 0.0
        self.be_stop_active = False
