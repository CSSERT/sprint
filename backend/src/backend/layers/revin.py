import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(
        self,
        n_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, n_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, n_features))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        self.center = (
            x[:, -1:, :] if self.subtract_last else x.mean(dim=1, keepdim=True)
        ).detach()
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x = (x - self.center) / self.std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps**2)
        return x * self.std.unsqueeze(2) + self.center.unsqueeze(2)
