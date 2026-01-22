import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
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
