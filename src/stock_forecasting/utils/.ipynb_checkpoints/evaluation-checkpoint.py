from typing import Literal

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def mean_absolute_scaled_error(y_true: np.ndarray, y_hat: np.ndarray, y_train: np.ndarray) -> float:
  d = np.abs(np.diff(y_train)).mean()
  mae = mean_absolute_error(y_true, y_hat)
  return mae / d

def directional_accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
  true_dir = np.sign(np.diff(y_true))
  hat_dir = np.sign(np.diff(y_hat))
  return (true_dir == hat_dir).mean()

def evaluate_model(
  model: nn.Module,
  test_loader: DataLoader,
  train_loader: DataLoader = None,
  device: Literal["cuda", "cpu"] = "cuda",
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float | None]]:
  model.to(device)
  
  model.eval()
  y_trues, y_hats = [], []
  with torch.no_grad():
    for X_batch, y_batch in test_loader:
      y_trues.append(np.atleast_1d(y_batch.squeeze().cpu().numpy()))
      y_hats.append(np.atleast_1d(model(X_batch.to(device)).squeeze().cpu().numpy()))
  y_true = np.concatenate(y_trues)
  y_hat = np.concatenate(y_hats)

  y_trains = []
  if train_loader is not None:
    for _, y_batch in train_loader:
      y_trains.append(np.atleast_1d(y_batch.squeeze().cpu().numpy()))
  y_train = np.concatenate(y_trains) if len(y_trains) else None
  
  mae = mean_absolute_error(y_true, y_hat)
  mase = (
    mean_absolute_scaled_error(y_true, y_hat, y_train)
    if y_train is not None else None
  )
  r2 = r2_score(y_true, y_hat)
  da = directional_accuracy(y_true, y_hat)

  metrics = {
    "MAE": mae,
    "MASE (%)": mase * 100 if mase is not None else None,
    "R2": r2,
    "Dir Accuracy": da,
  }

  return metrics, y_true, y_hat, y_train
