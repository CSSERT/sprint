import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
  def __init__(
    self,
    input_size: int,
    output_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = True,
  ) -> None:
    super().__init__()

    self.encoder = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout,
      bidirectional=bidirectional,
    )

    self.regressor = nn.Linear(
      hidden_size * (2 if bidirectional else 1),
      output_size,
    )

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    out, _ = self.encoder(X)
    out = out[:, -1, :]
    out = self.regressor(out)
    if self.regressor.out_features == 1:
      out = out.squeeze(1)
    return out