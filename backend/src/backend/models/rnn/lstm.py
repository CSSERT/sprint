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
        embed_dim: int = 8,
        n_layers: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.0,
        bidirectional: bool = True,
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
            bidirectional=bidirectional,
        )

        self.regressor = nn.Linear(
            hidden_size * (2 if bidirectional else 1),
            self.n_horizons * self.n_quantiles,
        )

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        b = x.size(0)

        out, _ = self.encoder(x)
        out = out[:, -1, :]

        out = self.regressor(out)
        out = out.view(b, self.n_horizons, self.n_quantiles)

        return out
