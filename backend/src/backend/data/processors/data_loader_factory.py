import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...types.data import (
    DataProcessor,
    DataState,
    TrainTestData,
    TrainTestLoader,
    WindowingMeta,
)


class DataLoaderFactory(DataProcessor):
    def __init__(self, target_col: str) -> None:
        self.target_col = target_col

    def _create_data_loader(
        self,
        df: pd.DataFrame,
        meta: WindowingMeta,
        batch_size: int = 32,
    ) -> DataLoader:
        x = torch.as_tensor(
            df.drop([self.target_col], axis=1).values.reshape(
                len(df), meta.n_lags, meta.n_features
            ),
            dtype=torch.float32,
        )
        y = torch.as_tensor(
            df[self.target_col].values,
            dtype=torch.float32,
        )

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def apply(
        self,
        data: DataState[None, TrainTestData, WindowingMeta],
    ) -> DataState[None, TrainTestLoader, None]:
        extras = TrainTestLoader(
            train=self._create_data_loader(data.extras.train, data.meta),
            test=self._create_data_loader(data.extras.test, data.meta),
        )
        return DataState(None, extras, meta=None)
