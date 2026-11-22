"""An order's life cycle, from placed to delivered."""

FLOW = ["placed", "cooking", "ready", "out_for_delivery", "delivered"]
CANCELLABLE = {"placed", "cooking"}


class InvalidTransition(ValueError):
    pass


class Order:
    def __init__(self, order_id: int):
        self.id = order_id
        self.status = "placed"
        self.history = ["placed"]

    def advance(self) -> str:
        """Move to the next step and return the new status."""
        if self.status not in FLOW or self.status == FLOW[-1]:
            raise InvalidTransition(f"order {self.id} can't advance from {self.status}")
        self.status = FLOW[FLOW.index(self.status) + 1]
        self.history.append(self.status)
        return self.status

    def cancel(self) -> None:
        if self.status not in CANCELLABLE:
            raise InvalidTransition(f"order {self.id} can't be cancelled once it is {self.status}")
        self.status = "cancelled"
        self.history.append("cancelled")
