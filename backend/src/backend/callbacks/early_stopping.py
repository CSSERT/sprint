class EarlyStopping:
    def __init__(
        self,
        *,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_loss: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = loss
            return

        if self.mode == "min":
            improved = loss < self.best_loss - self.min_delta
        else:
            improved = loss > self.best_loss + self.min_delta

        if improved:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self) -> None:
        self.best_loss = None
        self.counter = 0
        self.should_stop = False
