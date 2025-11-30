from pathlib import Path
from typing import Literal

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset

from stock_forecasting.utils import Cfg

class StockDataset(Dataset):
  def __init__(
    self,
    ticker: str,
    data_dir: Path | str = Cfg.DATA_DIR,
    interval: Literal["day", "week"] = Cfg.INTERVAL,
    feature_cols: list[str] | str = Cfg.FEATURE_COLS,
    target_col: str = Cfg.TARGET_COL,
    sliding_window_size: int = Cfg.SLIDING_WINDOW_SIZE,
    val_size_ratio: float = Cfg.VAL_SIZE_RATIO,
    test_size_ratio: float = Cfg.TEST_SIZE_RATIO,
    mode: Literal["train", "val", "test"] = "train",
    min_max_scale: bool = True,
  ) -> None:
    super().__init__()

    if isinstance(data_dir, str):
      data_dir = Path(data_dir)
    if isinstance(feature_cols, str):
      feature_cols = [feature_cols]

    file_path = data_dir / interval / f"{ticker}.csv"
    df = pd.read_csv(file_path, index_col="time", parse_dates=True)[
      ["open", "high", "low", "close", "volume"]
    ]

    df["volatility"] = (df["high"] - df["low"]) / df["close"]

    features = df[feature_cols]
    targets = df[target_col]

    num_samples = len(features)
    train_split_idx = int(num_samples * (1 - val_size_ratio - test_size_ratio))
    val_split_idx = int(num_samples * (1 - test_size_ratio))

    scaler = None
    if min_max_scale:
      scaler = MinMaxScaler()
      scaler.fit(features.iloc[:train_split_idx])
    
    F = torch.tensor(
      scaler.transform(features) if min_max_scale else features.values,
      dtype=torch.float32,
    )
    T = torch.tensor(targets.values, dtype=torch.float32)

    X = F.unfold(0, sliding_window_size, 1)[:len(T) - sliding_window_size]
    y = T[sliding_window_size:]

    if mode == "train":
      self.X = X[:train_split_idx]
      self.y = y[:train_split_idx]
    elif mode == "val":
      self.X = X[train_split_idx:val_split_idx]
      self.y = y[train_split_idx:val_split_idx]
    else:
      self.X = X[val_split_idx:]
      self.y = y[val_split_idx:]

  def __len__(self) -> int:
    return len(self.X)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.X[idx], self.y[idx]