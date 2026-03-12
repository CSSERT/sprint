import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData


class WalkForwardSplitter(DataProcessor):
    def __init__(
        self,
        *,
        n_folds: int,
        gap: int,
        test_size: int,
        min_train_size: int,
        expanding: bool,
    ) -> None:
        self.n_folds = n_folds
        self.gap = gap
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.expanding = expanding

    def apply(
        self,
        data: DataState[pd.DataFrame, None, None],
    ) -> DataState[None, list[TrainTestData], None]:
        if len(data._default) < self.min_train_size + self.gap + self.test_size:
            raise ValueError("Dataset too small for 1 fold")

        folds: list[TrainTestData] = []

        for i in range(self.n_folds - 1, -1, -1):
            test_end = len(data._default) - i * self.test_size
            test_start = test_end - self.test_size

            train_end = max(0, test_start - self.gap)
            train_start = (
                0 if self.expanding else max(0, train_end - self.min_train_size)
            )

            folds.append(
                TrainTestData(
                    train=data._default.iloc[train_start:train_end],
                    test=data._default.iloc[test_start:test_end],
                )
            )

        return DataState(None, folds, meta=None)
