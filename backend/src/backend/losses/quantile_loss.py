import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    quantiles: torch.Tensor

    def __init__(
        self,
        quantiles: list[float],
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles).view(1, 1, -1).to(device)
        )

    def forward(self, hat: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        diff = true.unsqueeze(-1) - hat
        loss = torch.maximum(self.quantiles * diff, (self.quantiles - 1) * diff)
        return loss.mean()
