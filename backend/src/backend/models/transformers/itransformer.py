import torch
import torch.nn as nn

from ...layers import PosEncType, PositionalEncoding, TSTEncoder
from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.transformers.itransformer")
class iTransformer(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        n_lags: int,
        horizons: list[int],
        quantiles: list[float],
        feature_target_idx: int,
        n_layers: int = 3,
        n_heads: int = 4,
        embed_dim: int = 128,
        ticker_embed_dim: int = 16,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        pos_enc_type: PosEncType | None = None,
        learnable_pos_enc: bool = True,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_tickers=n_tickers,
            horizons=horizons,
            quantiles=quantiles,
            embed_dim=ticker_embed_dim,
        )

        self.feature_target_idx = feature_target_idx
        self.n_features = n_features

        self.variate_embed = nn.Linear(n_lags, embed_dim)

        self.pos_enc = PositionalEncoding(
            seq_len=n_features + ticker_embed_dim,
            embed_dim=embed_dim,
            pos_enc_type=pos_enc_type,
            learnable=learnable_pos_enc,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList(
            [TSTEncoder(embed_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.head = nn.Linear(embed_dim, self.n_horizons * self.n_quantiles)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        B, T, F = x.shape
        x = x.permute(0, 2, 1)
        x = self.variate_embed(x)
        x = self.pos_enc(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.head(x)
        x = self.dropout(x)

        x = x.view(B, F, self.n_horizons, self.n_quantiles)
        x = x[:, : self.n_features]
        x = x.permute(0, 2, 3, 1)

        return x[:, :, :, self.feature_target_idx]
