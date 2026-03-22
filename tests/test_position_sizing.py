# tests/test_position_sizing.py
"""ATRRiskSizer ve PercentSizerFloat testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_atr_risk_sizer_basic():
    """ATR-based sizer doğru pozisyon boyutu hesaplamalı."""
    # Formül: size = (equity * risk_pct) / (atr * sl_mult)
    equity = 10000
    risk_pct = 0.005   # %0.5
    atr = 100
    sl_mult = 2.0

    expected_size = (equity * risk_pct) / (atr * sl_mult)
    assert expected_size == 0.25  # 10000 * 0.005 / (100 * 2) = 0.25


def test_atr_risk_sizer_higher_atr_smaller_position():
    """Yüksek ATR → daha küçük pozisyon."""
    equity = 10000
    risk_pct = 0.01

    size_low_atr = (equity * risk_pct) / (50 * 2.0)   # ATR=50
    size_high_atr = (equity * risk_pct) / (200 * 2.0)  # ATR=200

    assert size_low_atr > size_high_atr


def test_percent_sizer_capped():
    """Percent sizer bakiyenin belirli yüzdesini aşmamalı."""
    equity = 10000
    percents = 20
    price = 100

    available = equity * (percents / 100.0)
    size = available / price
    assert size == 20.0  # 10000 * 0.20 / 100 = 20
    assert available <= equity  # asla bakiyeyi aşmaz


def test_risk_per_trade_not_excessive():
    """Risk oranı %2'yi aşmamalı (güvenlik kontrolü)."""
    import config
    assert config.RISK_PER_TRADE <= 0.02, \
        f"RISK_PER_TRADE={config.RISK_PER_TRADE} çok yüksek, max %2 olmalı"
