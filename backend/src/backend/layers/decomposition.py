import torch
import torch.nn as nn


class Decomposition(nn.Module):
    def __init__(
        self,
        kernel_size: int = 25,
        mode: str = "avg",
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.mode = mode

        match mode:
            case "avg":
                self.get_trend = nn.AvgPool1d(
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            case "linear":
                self.get_trend = LinearTrend(
                    kernel_size=kernel_size,
                )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)

        trend = self.get_trend(x)
        residual = x - trend

        return trend.permute(0, 2, 1), residual.permute(0, 2, 1)


class LinearTrend(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.linear = nn.Linear(kernel_size, kernel_size)

    def forward(self, x):
        B, T, C = x.shape

        x = x.unfold(-1, self.kernel_size, 1)
        x = x.reshape(B, C, -1, self.kernel_size)

        trend = self.linear(x)
        return trend
