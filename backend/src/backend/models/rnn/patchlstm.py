import torch
import torch.nn as nn

from ...layers import Patching, PositionalEncoding, RevIN, PatchPadType, PosEncType
from ...types.model import QuantileModel
from ..registry import register_model


@register_model("sprint.rnn.patchlstm")
class PatchLSTM(QuantileModel):
    def __init__(
        self,
        *,
        n_features: int,
        n_tickers: int,
        n_lags: int,
        horizons: list[int],
        quantiles: list[float],
        feature_target_idx: int,
        patch_len: int = 16,
        stride: int = 16,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        ticker_embed_dim: int = 16,
        pos_enc_type: PosEncType | None = "normal",
        learnable_pos_enc: bool = True,
        patch_pad_type: PatchPadType | None = "end",
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

        self.revin = RevIN(n_features)

        n_patches = (n_lags - patch_len) // stride + 1
        self.patch = Patching(
            patch_len=patch_len,
            stride=stride,
            pad_type=patch_pad_type,
        )
        if patch_pad_type is not None:
            n_patches += 1

        self.features_proj = nn.Linear(patch_len, hidden_size)

        self.pos_enc = PositionalEncoding(
            seq_len=n_patches,
            embed_dim=hidden_size,
            pos_enc_type=pos_enc_type,
            learnable=learnable_pos_enc,
            dropout=dropout,
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(
            hidden_size * n_patches,
            self.n_horizons * self.n_quantiles,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ticker: torch.Tensor) -> torch.Tensor:
        x = self.revin.norm(x)

        x = super().forward(x, ticker)

        x = self.patch(x)
        x = self.features_proj(x)
        x = self.pos_enc(x)

        B, F, N, D = x.shape
        x = x.reshape(B * F, N, D)

        x, _ = self.encoder(x)

        x = x.reshape(B, F, N, D)
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)

        x = x.view(B, F, self.n_horizons, self.n_quantiles)
        x = x[:, : self.n_features]
        x = x.permute(0, 2, 3, 1)

        x = self.revin.denorm(x)

        return x[:, :, :, self.feature_target_idx]
