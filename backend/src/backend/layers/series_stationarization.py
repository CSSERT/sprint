import torch
import torch.nn as nn


class SeriesStationarization(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

        self.mu = nn.Parameter(torch.zeros(1, 1, num_features))
        self.sigma = nn.Parameter(torch.ones(1, 1, num_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, F = x.shape

        x_mean = x.mean(dim=1, keepdim=True).detach()
        x_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        self.mu.data = x_mean.squeeze(1)
        self.sigma.data = x_std.squeeze(1)

        x_stationary = (x - x_mean) / x_std

        return x_stationary, x_mean, x_std
