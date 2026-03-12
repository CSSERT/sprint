import math as m
from typing import Literal

import torch
import torch.nn as nn

type PosEncType = Literal["normal", "sincos"]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        pos_enc_type: PosEncType | None,
        learnable: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        match pos_enc_type:
            case None:
                pos_emb = torch.empty((seq_len, embed_dim))
                nn.init.uniform_(pos_emb, -0.02, 0.02)
                learnable = False
            case "normal":
                pos_emb = torch.zeros((seq_len, 1))
                nn.init.normal_(pos_emb, mean=0.0, std=0.1)
            case "sincos":
                pos_emb = torch.zeros(seq_len, embed_dim)

                pos = torch.arange(seq_len).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, embed_dim, 2) * -(m.log(10_000.0) / embed_dim)
                )

                pos_emb[:, 0::2] = torch.sin(pos * div)
                pos_emb[:, 1::2] = torch.cos(pos * div)

                pos_emb -= pos_emb.mean()
                pos_emb /= pos_emb.std() * 10

        self.pos_emb = nn.Parameter(pos_emb, requires_grad=learnable)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_emb[: x.size(2)]
        x = x + pos.unsqueeze(0).unsqueeze(0)
        return self.dropout(x)
