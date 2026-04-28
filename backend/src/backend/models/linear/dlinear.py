import torch
import torch.nn as nn

from ...layers import Decomposition
from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.linear.dlinear")
class DLinear(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        n_lags: int,
        horizons: list[int],
        quantiles: list[float],
        feature_target_idx: int,
        kernel_size: int = 25,
        decomp_mode: str = "avg",
        embed_dim: int = 16,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_tickers=n_tickers,
            horizons=horizons,
            quantiles=quantiles,
            embed_dim=embed_dim,
        )

        self.feature_target_idx = feature_target_idx
        self.n_features = n_features
        self.n_lags = n_lags

        self.decomposition = Decomposition(kernel_size=kernel_size, mode=decomp_mode)

        self.trend_linear = nn.Linear(n_lags, n_lags)
        self.residual_linear = nn.Linear(n_lags, n_lags)
        self.head = nn.Linear(n_lags, self.n_horizons * self.n_quantiles)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        B, T, F = x.shape

        x = x.permute(0, 2, 1)

        trend, residual = self.decomposition(x)

        trend = trend.permute(0, 2, 1).reshape(B * F, T)
        residual = residual.permute(0, 2, 1).reshape(B * F, T)

        trend = self.trend_linear(trend)
        residual = self.residual_linear(residual)

        x = trend + residual

        x = self.head(x)

        x = x.reshape(B, F, self.n_horizons, self.n_quantiles)
        x = x[:, : self.n_features]
        x = x.permute(0, 2, 3, 1)

        return x[:, :, :, self.feature_target_idx]
