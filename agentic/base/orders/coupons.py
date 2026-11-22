"""Coupon codes customers can apply at checkout."""

COUPONS = {
    # code: (kind, value, minimum order in rupees, cap in rupees or None)
    "WELCOME50": ("flat", 50, 200, None),
    "FEAST10": ("percent", 10, 300, 100),
}


def coupon_discount(code: str, amount: int) -> int:
    """Rupees taken off an order worth `amount` (after other discounts).

    Unknown codes raise ValueError. An order below the coupon's minimum gets no discount.
    Percentage coupons round to the nearest rupee and never exceed their cap.
    """
    if code not in COUPONS:
        raise ValueError(f"unknown coupon: {code}")
    kind, value, minimum, cap = COUPONS[code]
    if amount < minimum:
        return 0
    if kind == "flat":
        return min(value, amount)
    off = round(amount * value / 100)
    return min(off, cap) if cap is not None else off
