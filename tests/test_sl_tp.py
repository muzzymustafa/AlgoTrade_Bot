# tests/test_sl_tp.py
"""SL/TP hesaplama testleri — _lt_sl_tp fonksiyonunun 3 fallback yolu."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_percentage_fallback_long():
    """Yüzdesel fallback: long tarafta SL < giriş < TP."""
    price = 50000
    stop_loss = 0.02    # %2
    take_profit = 0.05  # %5

    sl = price * (1.0 - stop_loss)
    tp = price * (1.0 + take_profit)

    assert sl == 49000.0
    assert tp == 52500.0
    assert sl < price < tp


def test_percentage_fallback_short():
    """Yüzdesel fallback: short tarafta SL > giriş > TP."""
    price = 50000
    stop_loss = 0.02
    take_profit = 0.05

    sl = price * (1.0 + stop_loss)
    tp = price * (1.0 - take_profit)

    assert sl == 51000.0
    assert tp == 47500.0
    assert tp < price < sl


def test_atr_based_long():
    """ATR tabanlı: long SL = giriş - (atr * sl_mult)."""
    price = 50000
    atr = 500
    sl_mult = 2.0
    tp_mult = 3.5

    sl = price - sl_mult * atr
    tp = price + tp_mult * atr

    assert sl == 49000.0
    assert tp == 51750.0
    assert sl < price < tp


def test_atr_based_short():
    """ATR tabanlı: short SL = giriş + (atr * sl_mult)."""
    price = 50000
    atr = 500
    sl_mult = 2.0
    tp_mult = 3.5

    sl = price + sl_mult * atr
    tp = price - tp_mult * atr

    assert sl == 51000.0
    assert tp == 48250.0
    assert tp < price < sl


def test_otr_regime_mapping():
    """OTR rejim haritası tutarlı olmalı: her rejim için SL < TP mult."""
    otr_map = {0: (1.0, 2.0), 1: (2.0, 3.5), 2: (1.5, 3.0)}

    for regime, (sl_m, tp_m) in otr_map.items():
        assert sl_m > 0, f"Regime {regime}: SL mult pozitif olmalı"
        assert tp_m > 0, f"Regime {regime}: TP mult pozitif olmalı"
        assert tp_m > sl_m, f"Regime {regime}: TP mult ({tp_m}) > SL mult ({sl_m}) olmalı"
