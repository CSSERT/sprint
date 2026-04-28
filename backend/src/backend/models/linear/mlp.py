import torch
import torch.nn as nn

from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.linear.mlp")
class MultiLayerPerceptron(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        n_lags: int,
        horizons: list[int],
        quantiles: list[float],
        feature_target_idx: int,
        hidden_size: int = 256,
        embed_dim: int = 16,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_tickers=n_tickers,
            horizons=horizons,
            quantiles=quantiles,
            embed_dim=embed_dim,
        )

        self.n_features = n_features
        self.feature_target_idx = feature_target_idx

        self.net = nn.Sequential(
            nn.Linear(n_lags, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, self.n_horizons * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        B, T, F = x.shape
        x = x.reshape(B * F, T)

        x = self.net(x)

        x = x.reshape(B, F, self.n_horizons, self.n_quantiles)
        x = x[:, : self.n_features]
        x = x.permute(0, 2, 3, 1)

        return x[:, :, :, self.feature_target_idx]
