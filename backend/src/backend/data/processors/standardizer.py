from dataclasses import dataclass

import pandas as pd

from ...types.data import DataProcessor, DataState


@dataclass
class TrainTestData:
    train: pd.DataFrame
    test: pd.DataFrame


class Standardizer(DataProcessor):
    def prepare(self, data: DataState[None, TrainTestData]) -> None:
        self.mean = data.extras.train.mean()
        self.std = data.extras.train.std()

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def apply(
        self, data: DataState[None, TrainTestData]
    ) -> DataState[None, TrainTestData]:
        train_applied = self._scale(data.extras.train)
        test_applied = self._scale(data.extras.test)

        return DataState(None, TrainTestData(train=train_applied, test=test_applied))
