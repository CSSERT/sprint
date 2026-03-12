import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...types.data import (
    DataProcessor,
    DataState,
    FactoryMeta,
    TrainTestData,
    TrainTestLoader,
    WindowMeta,
)


class DataLoaderFactory(DataProcessor):
    def __init__(self, *, ticker_col: str) -> None:
        self.ticker_col = ticker_col

    def _create_data_loader(
        self,
        df: pd.DataFrame,
        meta: WindowMeta,
        batch_size: int = 32,
    ) -> DataLoader:
        x = torch.as_tensor(
            df.loc[:, meta.feature_columns].values.reshape(
                len(df), meta.n_lags, len(meta.feature_columns) // meta.n_lags
            ),
            dtype=torch.float32,
        )
        y = torch.as_tensor(
            df.loc[:, meta.target_columns].values,
            dtype=torch.float32,
        )
        ticker = torch.tensor(
            df.loc[:, [self.ticker_col]].values,
            dtype=torch.long,
        )

        dataset = TensorDataset(x, y, ticker)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def apply(
        self,
        data: DataState[None, TrainTestData, WindowMeta],
    ) -> DataState[None, TrainTestLoader, FactoryMeta]:
        extras = TrainTestLoader(
            train=self._create_data_loader(data.extras.train, data.meta),
            test=self._create_data_loader(data.extras.test, data.meta),
        )
        meta = FactoryMeta(feature_target_idx=data.meta.feature_target_idx)
        return DataState(None, extras, meta=meta)
