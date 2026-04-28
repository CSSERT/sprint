import torch
import torch.nn as nn


class DestationaryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.scale = nn.Parameter(
            torch.tensor(self.head_dim**-0.5), requires_grad=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        tau = 1.0
        delta = torch.zeros_like(x_mean)

        attn = attn / (tau * x_std.unsqueeze(-1)) + delta.unsqueeze(-1)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return out
