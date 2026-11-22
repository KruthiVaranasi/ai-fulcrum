"""The owner's daily summary."""


def summary(orders: list[dict]) -> dict:
    """Summarise a day of orders.

    Each order is a dict with: total (rupees), status ("delivered" or "cancelled"),
    promised_min (ETA given to the customer) and actual_min (actual delivery time, delivered orders only).
    """
    delivered = [o for o in orders if o["status"] == "delivered"]
    late = [o for o in delivered if o["actual_min"] > o["promised_min"]]
    return {
        "orders": len(delivered),
        "revenue": sum(o["total"] for o in delivered),
        "cancelled": sum(o["status"] == "cancelled" for o in orders),
        "early": sum(o["actual_min"] < o["promised_min"] for o in delivered),
        "on_time": sum(o["actual_min"] == o["promised_min"] for o in delivered),
        "late": len(late),
        "average_delay": round(sum(o["actual_min"] - o["promised_min"] for o in late) / len(late), 1) if late else 0.0,
    }
