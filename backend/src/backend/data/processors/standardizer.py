from typing import Any

import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData


class Standardizer(DataProcessor):
    def __init__(self, *, scale_cols: list[str]) -> None:
        self.scale_cols = scale_cols

    def prepare(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> None:
        train = data.extras.train.loc[:, self.scale_cols]
        self.mean = train.mean()
        self.std = train.std().replace(0, 1)

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**(df.loc[:, self.scale_cols] - self.mean) / self.std)

    def _inverse_scale(self, scaled_value: Any) -> Any:
        return scaled_value * self.std["close"] + self.mean["close"]

    def apply(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> DataState[None, TrainTestData, None]:
        extras = TrainTestData(
            train=self._scale(data.extras.train),
            test=self._scale(data.extras.test),
        )
        return DataState(None, extras, meta=None)
