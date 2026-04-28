import torch
import torch.nn as nn

from ...layers import PosEncType, PositionalEncoding, TSTEncoder
from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.transformers.vanilla")
class VanillaTransformer(QuantileModel):
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
        n_heads: int = 16,
        embed_dim: int = 128,
        ticker_embed_dim: int = 16,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        pos_enc_type: PosEncType | None = "normal",
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

        self.features_proj = nn.Linear(1, embed_dim)

        self.pos_enc = PositionalEncoding(
            seq_len=n_lags,
            embed_dim=embed_dim,
            pos_enc_type=pos_enc_type,
            learnable=learnable_pos_enc,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList(
            [TSTEncoder(embed_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(
            embed_dim * n_lags,
            self.n_horizons * self.n_quantiles,
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, ticker)

        B, T, F = x.shape
        x = x.reshape(B * F, T, 1)
        x = self.features_proj(x)
        x = self.pos_enc(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)

        x = x.view(B, F, self.n_horizons, self.n_quantiles)
        x = x[:, : self.n_features]
        x = x.permute(0, 2, 3, 1)

        return x[:, :, :, self.feature_target_idx]
