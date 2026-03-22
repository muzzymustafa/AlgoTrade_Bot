# tests/test_live.py
"""Live/Paper trading modülü testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_paper_broker_open_close_long():
    """Paper broker: long aç, kapat, PnL doğru."""
    from live.paper_broker import PaperBroker

    broker = PaperBroker(cash=10000, commission=0.001)

    # Long aç
    broker.execute_fill("BTC/USDT", "buy", 0.1, 50000)
    assert broker.position is not None
    assert broker.position["side"] == "long"

    # Long kapat
    broker.execute_fill("BTC/USDT", "sell", 0.1, 51000)
    assert broker.position is None

    # PnL: (51000-50000)*0.1 - komisyon
    assert broker.cash > 10000  # kârlı
    assert len(broker.trade_log) == 1


def test_paper_broker_stop_trigger():
    """Paper broker: stop emri bar low'u ile tetiklenmeli."""
    from live.paper_broker import PaperBroker

    broker = PaperBroker(cash=10000, commission=0.001)
    broker.execute_fill("BTC/USDT", "buy", 0.1, 50000)

    # SL emri
    broker.place_stop_order("BTC/USDT", "sell", 0.1, 49000)
    assert len(broker.pending_orders) == 1

    # Bar low SL'yi tetiklemiyor
    broker.check_pending_orders("BTC/USDT", {"high": 50500, "low": 49500})
    assert broker.position is not None  # hala açık

    # Bar low SL'yi tetikliyor
    broker.check_pending_orders("BTC/USDT", {"high": 49500, "low": 48500})
    assert broker.position is None  # SL tetiklendi


def test_paper_broker_limit_trigger():
    """Paper broker: limit emri (TP) bar high ile tetiklenmeli."""
    from live.paper_broker import PaperBroker

    broker = PaperBroker(cash=10000, commission=0.001)
    broker.execute_fill("BTC/USDT", "buy", 0.1, 50000)

    # TP emri
    broker.place_limit_order("BTC/USDT", "sell", 0.1, 52000)

    # Bar high TP'ye ulaşmıyor
    broker.check_pending_orders("BTC/USDT", {"high": 51500, "low": 50500})
    assert broker.position is not None

    # Bar high TP'yi tetikliyor
    broker.check_pending_orders("BTC/USDT", {"high": 52500, "low": 51500})
    assert broker.position is None


def test_state_save_load(tmp_path):
    """State JSON kaydedilip yüklenebilmeli."""
    from live.state import State

    state = State(cash=15000, trade_count=5, last_bar_time="2025-01-01T00:00:00")
    path = str(tmp_path / "test_state.json")
    state.save(path)

    loaded = State.load(path)
    assert loaded.cash == 15000
    assert loaded.trade_count == 5
    assert loaded.last_bar_time == "2025-01-01T00:00:00"


def test_state_load_missing():
    """Dosya yoksa varsayılan state dönmeli."""
    from live.state import State
    state = State.load("nonexistent_path_12345.json")
    assert state.cash == 10000
    assert state.trade_count == 0


def test_signal_generator_no_crash():
    """SignalGenerator boş veriyle crash olmamalı."""
    import pandas as pd
    from live.signal_generator import SignalGenerator

    sg = SignalGenerator()
    df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    indicators = sg.compute_indicators(df_empty, df_empty)
    sig = sg.get_signal(indicators, None, 100)
    assert sig.direction is None


def test_notifier_noop():
    """Notifier no-op çağrılabilmeli."""
    from live.notifier import Notifier
    n = Notifier()
    n.send("test")  # hata vermemeli
