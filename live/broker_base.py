# live/broker_base.py
"""Broker soyut arayüzü — paper ve live broker'ların uygulaması gereken metotlar."""
from abc import ABC, abstractmethod


class BrokerBase(ABC):

    @abstractmethod
    def get_balance(self) -> float:
        """Kullanılabilir nakit bakiye."""
        ...

    @abstractmethod
    def get_equity(self, current_price: float) -> float:
        """Nakit + pozisyon değeri."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> dict | None:
        """
        Açık pozisyon varsa:
          {'side': 'long'|'short', 'size': float, 'entry_price': float}
        Yoksa None.
        """
        ...

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, size: float) -> dict:
        """Market emir. Returns: {'order_id': str, 'filled_price': float, ...}"""
        ...

    @abstractmethod
    def place_stop_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        """Stop-loss emri."""
        ...

    @abstractmethod
    def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> dict:
        """Take-profit limit emri."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Tek emir iptal."""
        ...

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> int:
        """Tüm açık emirleri iptal. Kaç emir iptal edildi döndür."""
        ...

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list:
        """Açık emirlerin listesi."""
        ...
