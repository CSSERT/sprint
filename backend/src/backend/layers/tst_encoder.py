import torch
import torch.nn as nn


class TSTEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int = 256,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.pre_norm = pre_norm

        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_attn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm_attn(x)
            attn_out, _ = self.attn(x, prev_attn)
            x = x + self.dropout_attn(attn_out)

            x = self.norm_ff(x)
            ff_out = self.ff(x)
            x = x + self.dropout_ff(ff_out)
        else:
            attn_out, _ = self.attn(x, prev_attn)
            x = x + self.dropout_attn(attn_out)
            x = self.norm_attn(x)

            ff_out = self.ff(x)
            x = x + self.dropout_ff(ff_out)
            x = self.norm_ff(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        lsa: bool = True,
    ):
        super().__init__()

        head_dim = embed_dim // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prev_attn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if prev_attn is not None:
            attn = attn + prev_attn

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        return attn @ v, attn


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        def proj() -> nn.Module:
            return nn.Linear(embed_dim, embed_dim)

        self.q_proj = proj()
        self.k_proj = proj()
        self.v_proj = proj()

        self.attn = ScaledDotProductAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )

        self.out_proj = nn.Sequential(proj(), nn.Dropout(proj_dropout))

    def forward(
        self,
        x: torch.Tensor,
        prev_attn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(
                1, 2
            )

        q = reshape(self.q_proj(x))
        k = reshape(self.k_proj(x))
        v = reshape(self.v_proj(x))

        attn_out, attn = self.attn(q, k, v, prev_attn)

        attn_out = attn_out.transpose(1, 2).reshape(*x.shape)
        attn_out = self.out_proj(attn_out)

        return attn_out, attn
