import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData


class TrainTestSplitter(DataProcessor):
    def __init__(self, *, test_size: float):
        self.test_size = test_size

    def apply(
        self,
        data: DataState[pd.DataFrame, None, None],
    ) -> DataState[None, TrainTestData, None]:
        n_samples = len(data._default)
        split = int(n_samples * (1 - self.test_size))

        extras = TrainTestData(
            train=data._default.iloc[0:split],
            test=data._default.iloc[split:],
        )
        return DataState(None, extras, meta=None)
