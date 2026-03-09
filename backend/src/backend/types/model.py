import torch
import torch.nn as nn


class QuantileModel(nn.Module):
    horizons: torch.Tensor
    quantiles: torch.Tensor

    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        horizons: list[int],
        quantiles: list[float],
        embed_dim: int = 8,
    ) -> None:
        super().__init__()

        self.n_horizons = len(horizons)
        self.n_quantiles = len(quantiles)
        self.input_size = n_features + embed_dim

        self.register_buffer(
            "horizons",
            torch.tensor(horizons, dtype=torch.long),
        )
        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles, dtype=torch.float32),
        )

        self.emb = nn.Embedding(n_tickers, embed_dim)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        emb = self.emb(ticker.squeeze(-1)).unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.cat([x, emb], dim=-1)
