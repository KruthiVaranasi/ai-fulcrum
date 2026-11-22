"""Delivery fee and the ETA promised to the customer."""

import math

PEAK_HOURS = [(12, 15), (19, 22)]  # [start, end) in 24-hour time: lunch and dinner rush


def fee(distance_km: float) -> int:
    """Free up to 3 km, ₹20 up to 7 km, then ₹8 for every started km beyond 7."""
    if distance_km < 0:
        raise ValueError("distance can't be negative")
    if distance_km <= 3:
        return 0
    if distance_km <= 7:
        return 20
    return 20 + 8 * math.ceil(distance_km - 7)


def is_peak(hour: int) -> bool:
    return any(start <= hour < end for start, end in PEAK_HOURS)


def eta_minutes(distance_km: float, hour: int) -> int:
    """15 minutes to prepare, 4 minutes per started km, plus 10 minutes in peak hours."""
    minutes = 15 + 4 * math.ceil(distance_km)
    if is_peak(hour):
        minutes += 10
    return minutes
