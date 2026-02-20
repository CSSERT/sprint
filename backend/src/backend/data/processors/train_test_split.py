from dataclasses import dataclass

import pandas as pd

from ...types.data import DataProcessor, DataState


@dataclass
class TrainTestData:
    train: pd.DataFrame
    test: pd.DataFrame


class TrainTestSplitter(DataProcessor):
    def __init__(self, *, test_size: float):
        self.test_size = test_size

    def apply(
        self, data: DataState[pd.DataFrame, None]
    ) -> DataState[None, TrainTestData]:
        n_samples = len(data._default)
        split = int(n_samples * (1 - self.test_size))

        train = data._default.iloc[0:split]
        test = data._default.iloc[split:]

        return DataState(None, TrainTestData(train=train, test=test))
