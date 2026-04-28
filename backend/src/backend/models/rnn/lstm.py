import torch
import torch.nn as nn

from ...layers import RevIN
from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.rnn.lstm")
class LSTM(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        n_lags: int,
        horizons: list[int],
        quantiles: list[float],
        feature_target_idx: int,
        embed_dim: int = 16,
        hidden_size: int = 192,
        n_layers: int = 2,
        dropout: float = 0.2,
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

        self.revin = RevIN(n_features)

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(
                hidden_size // 2, self.n_features * self.n_horizons * self.n_quantiles
            ),
        )

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = self.revin.norm(x)

        x = super().forward(x, ticker)

        B = x.size(0)

        x, _ = self.encoder(x)
        x = x[:, -1, :]

        x = self.regressor(x)
        x = x.view(B, self.n_features, self.n_horizons, self.n_quantiles)
        x = x.permute(0, 2, 3, 1)

        x = self.revin.denorm(x)

        return x[:, :, :, self.feature_target_idx]
