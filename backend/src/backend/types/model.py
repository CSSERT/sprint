import torch
import torch.nn as nn


class QuantileModel(nn.Module):
    quantiles: torch.Tensor

    def __init__(
        self,
        *,
        n_features: int,
        n_horizons: int,
        quantiles: list[float],
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_horizons = n_horizons
        self.n_quantiles = len(quantiles)

        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles, dtype=torch.float32),
        )
