import torch
import torch.nn as nn

class BaselineCNN1D(nn.Module):
  def __init__(
    self,
    input_size: int,
    output_size: int,
    hidden_channels: int = 128,
    kernel_size: int = 3,
    padding: int = 1,
  ) -> None:
    super().__init__()
    
    self.conv1 = nn.Conv1d(
      in_channels=input_size,
      out_channels=hidden_channels,
      kernel_size=kernel_size,
      padding=padding,
    )
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv1d(
      in_channels=hidden_channels,
      out_channels=hidden_channels,
      kernel_size=kernel_size,
      padding=padding,
    )
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(hidden_channels, output_size)

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    out = X.permute(0, 2, 1)
    out = self.relu(self.conv1(out))
    out = self.relu(self.conv2(out))
    out = self.pool(out).squeeze(-1)
    out = self.fc(out)
    return out
