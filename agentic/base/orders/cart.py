"""A customer's cart."""

from .coupons import coupon_discount
from .menu import price


class Cart:
    def __init__(self):
        self.lines: dict[str, int] = {}

    def add(self, item: str, qty: int = 1) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        price(item)  # rejects items that aren't on the menu
        self.lines[item] = self.lines.get(item, 0) + qty

    def remove(self, item: str, qty: int = 1) -> None:
        if item not in self.lines:
            raise KeyError(item)
        left = self.lines[item] - qty
        if left > 0:
            self.lines[item] = left
        else:
            del self.lines[item]

    def subtotal(self) -> int:
        return sum(price(item) * qty for item, qty in self.lines.items())

    def bulk_discount(self) -> int:
        """10% off every line where the customer orders 5 or more of the same item."""
        return sum(round(price(item) * qty * 0.10) for item, qty in self.lines.items() if qty >= 5)

    def total(self, coupon: str | None = None) -> int:
        amount = self.subtotal() - self.bulk_discount()
        if coupon:
            amount -= coupon_discount(coupon, amount)
        return max(amount, 0)
