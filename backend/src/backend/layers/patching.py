from typing import Literal

import torch
import torch.nn as nn

type PatchPadType = Literal["end"]


class Patching(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
        pad_type: PatchPadType | None = "end",
    ) -> None:
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        if pad_type == "end":
            self.pad = nn.ReplicationPad1d((0, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        if self.pad is not None:
            x = self.pad(x)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
