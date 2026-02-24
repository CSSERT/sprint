import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData


class Standardizer(DataProcessor):
    def prepare(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> None:
        self.mean = data.extras.train.mean()
        self.std = data.extras.train.std()

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def apply(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> DataState[None, TrainTestData, None]:
        extras = TrainTestData(
            train=self._scale(data.extras.train),
            test=self._scale(data.extras.test),
        )
        return DataState(None, extras, meta=None)
