import pytest

from orders.cart import Cart
from orders.coupons import coupon_discount
from orders.delivery import eta_minutes, fee, is_peak
from orders.menu import price
from orders.report import summary
from orders.status import InvalidTransition, Order


# ---- menu and cart ----

def test_price_in_rupees():
    assert price("dosa") == 60
    assert price("chai") == 20
    with pytest.raises(KeyError):
        price("pizza")


def test_cart_totals():
    c = Cart()
    c.add("dosa", 2)
    c.add("chai")
    assert c.subtotal() == 140
    assert c.total() == 140


def test_bulk_discount_applies_per_line_from_five():
    c = Cart()
    c.add("dosa", 6)      # 360, 10% off = 36
    c.add("chai", 4)      # 80, below 5: no discount
    assert c.bulk_discount() == 36
    assert c.total() == 404
    c.add("chai")         # now 5 chai: 100, 10% off = 10
    assert c.bulk_discount() == 46


def test_cart_rejects_bad_input():
    c = Cart()
    with pytest.raises(ValueError):
        c.add("idli", 0)
    with pytest.raises(KeyError):
        c.add("pizza")
    c.add("idli", 3)
    c.remove("idli", 3)
    assert c.lines == {}


# ---- coupons ----

def test_flat_coupon():
    assert coupon_discount("WELCOME50", 250) == 50
    assert coupon_discount("WELCOME50", 199) == 0


def test_percent_coupon_is_capped():
    assert coupon_discount("FEAST10", 450) == 45
    assert coupon_discount("FEAST10", 2000) == 100
    assert coupon_discount("FEAST10", 299) == 0


def test_unknown_coupon():
    with pytest.raises(ValueError):
        coupon_discount("FREEFOOD", 500)


def test_cart_total_with_coupon():
    c = Cart()
    c.add("thali", 2)                         # 300
    assert c.total(coupon="WELCOME50") == 250
    assert c.total(coupon="FEAST10") == 270
    c2 = Cart()
    c2.add("biryani", 5)                      # 1100, bulk 110 -> 990
    assert c2.total(coupon="FEAST10") == 891  # 10% of 990 = 99


# ---- delivery ----

def test_fee_bands():
    assert fee(2.5) == 0
    assert fee(3) == 0
    assert fee(5) == 20
    assert fee(7) == 20
    assert fee(7.2) == 28
    assert fee(10) == 44


def test_peak_hours_boundaries():
    assert not is_peak(11)
    assert is_peak(12)
    assert is_peak(14)
    assert not is_peak(15)
    assert is_peak(19)
    assert is_peak(21)
    assert not is_peak(22)


def test_eta():
    assert eta_minutes(2, 10) == 23
    assert eta_minutes(2.1, 13) == 37
    assert eta_minutes(5, 21) == 45


# ---- report ----

ORDERS = [
    {"total": 300, "status": "delivered", "promised_min": 30, "actual_min": 25},
    {"total": 150, "status": "delivered", "promised_min": 30, "actual_min": 30},
    {"total": 220, "status": "delivered", "promised_min": 40, "actual_min": 52},
    {"total": 90, "status": "delivered", "promised_min": 35, "actual_min": 38},
    {"total": 400, "status": "cancelled", "promised_min": 30, "actual_min": None},
]


def test_summary_counts_and_revenue():
    s = summary(ORDERS)
    assert s["orders"] == 4
    assert s["revenue"] == 760
    assert s["cancelled"] == 1


def test_summary_delivery_against_eta():
    s = summary(ORDERS)
    assert (s["early"], s["on_time"], s["late"]) == (1, 1, 2)
    assert s["average_delay"] == 7.5
    assert summary([])["average_delay"] == 0.0


# ---- order status ----

def test_order_flow():
    o = Order(1)
    assert [o.advance() for _ in range(4)] == ["cooking", "ready", "out_for_delivery", "delivered"]
    assert o.history == ["placed", "cooking", "ready", "out_for_delivery", "delivered"]
    with pytest.raises(InvalidTransition):
        o.advance()


def test_order_cancel_rules():
    o = Order(2)
    o.advance()
    o.cancel()
    assert o.status == "cancelled"
    with pytest.raises(InvalidTransition):
        o.advance()
    late = Order(3)
    late.advance(); late.advance()
    with pytest.raises(InvalidTransition):
        late.cancel()
    assert issubclass(InvalidTransition, ValueError)
