import torch
import torch.nn as nn

from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.rnn.lstm")
class LSTM(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        horizons: list[int],
        quantiles: list[float],
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
            nn.Linear(hidden_size // 2, self.n_horizons * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        batch_size = x.size(0)

        out, _ = self.encoder(x)
        out = out[:, -1, :]

        out = self.regressor(out)
        out = out.view(batch_size, self.n_horizons, self.n_quantiles)

        return out
