"""The menu. Prices are whole rupees."""

MENU = {
    "idli": 40,
    "dosa": 60,
    "thali": 150,
    "biryani": 220,
    "samosa": 15,
    "chai": 20,
}


def price(item: str) -> int:
    """Price of one portion, in rupees."""
    if item not in MENU:
        raise KeyError(f"unknown item: {item}")
    return MENU[item]
