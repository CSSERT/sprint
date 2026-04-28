import torch
import torch.nn as nn

from . import DestationaryAttention


class NSTEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attn = DestationaryAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout_attn(self.attn(self.norm_attn(x), x_mean, x_std))
        x = x + self.dropout_ff(self.ff(self.norm_ff(x)))
        return x
