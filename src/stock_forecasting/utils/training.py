from typing import Literal, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from tqdm import tqdm

from stock_forecasting.utils import Cfg

def train_one_epoch(
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  model: nn.Module,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
) -> float:
  model.train()
  optimizer.zero_grad()
  
  y_hat = model(X_train)
  loss = criterion(y_hat, y_train)
  
  loss.backward()
  optimizer.step()
  
  return loss.item()
  
def train_model(
  train_loader: DataLoader,
  val_loader: DataLoader,
  model: nn.Module,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
  epochs: int = 10,
  on_each_epoch: Callable[[int, float, float], None] = None,
  device: Literal["cuda", "cpu"] = "cuda",
) -> tuple[list[float], list[float]]:
  model.to(device)
  
  train_losses = []
  val_losses = []
  
  for epoch in range(epochs):
    running_train_loss = 0.0
    for X_batch, y_batch in train_loader:
      running_train_loss += train_one_epoch(
        X_train=X_batch.to(device),
        y_train=y_batch.to(device),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
      )
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
      for X_batch, y_batch in val_loader:
        y_hat = model(X_batch.to(device))
        running_val_loss += criterion(y_hat, y_batch.to(device)).item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if on_each_epoch:
      on_each_epoch(epoch, avg_train_loss, avg_val_loss)

  return train_losses, val_losses