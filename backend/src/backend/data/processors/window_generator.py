from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ...types.data import DataProcessor, DataState


@dataclass
class TrainTestData:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TrainTestLoader:
    train: DataLoader
    test: DataLoader


class WindowedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        window_size: int,
    ) -> None:
        self.x = df.loc[:, feature_cols].values
        self.y = df.loc[:, target_col].values
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.x) - self.window_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        split = index + self.window_size
        return (
            torch.tensor(self.x[index:split], dtype=torch.float32),
            torch.tensor(self.y[split], dtype=torch.float32),
        )


class WindowGenerator(DataProcessor):
    def __init__(
        self,
        *,
        feature_cols: list[str],
        target_col: str,
        window_size: int,
        batch_size: int = 32,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.batch_size = batch_size

    def apply(
        self,
        data: DataState[None, TrainTestData],
    ) -> DataState[None, TrainTestLoader]:
        train_dataset = WindowedDataset(
            data.extras.train,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            window_size=self.window_size,
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
        )

        test_dataset = WindowedDataset(
            data.extras.test,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            window_size=self.window_size,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
        )

        return DataState(None, TrainTestLoader(train=train_loader, test=test_loader))
