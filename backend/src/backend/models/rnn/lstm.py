import torch
import torch.nn as nn

from ..registry import register_model


@register_model("sprint.rnn.lstm")
class LSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_quantiles: int,
        *,
        n_layers: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.regressor = nn.Linear(
            hidden_size * (2 if bidirectional else 1),
            n_horizons * n_quantiles,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        out, _ = self.encoder(x)
        out = out[:, -1, :]

        out = self.regressor(out)
        out = out.view(b, self.n_horizons, self.n_quantiles)

        return out
