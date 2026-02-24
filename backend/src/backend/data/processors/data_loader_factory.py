import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...types.data import (
    DataProcessor,
    DataState,
    TrainTestData,
    TrainTestLoader,
    WindowMeta,
)


class DataLoaderFactory(DataProcessor):
    def _create_data_loader(
        self,
        df: pd.DataFrame,
        meta: WindowMeta,
        batch_size: int = 32,
    ) -> DataLoader:
        x = torch.as_tensor(
            df.loc[:, meta.feature_columns].values.reshape(
                len(df), meta.n_lags, meta.n_features
            ),
            dtype=torch.float32,
        )
        y = torch.as_tensor(
            df.loc[:, meta.target_columns].values,
            dtype=torch.float32,
        )

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def apply(
        self,
        data: DataState[None, TrainTestData, WindowMeta],
    ) -> DataState[None, TrainTestLoader, None]:
        extras = TrainTestLoader(
            train=self._create_data_loader(data.extras.train, data.meta),
            test=self._create_data_loader(data.extras.test, data.meta),
        )
        return DataState(None, extras, meta=None)
