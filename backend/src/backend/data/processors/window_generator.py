import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData, WindowingMeta


class WindowGenerator(DataProcessor):
    def __init__(self, *, lags: list[int], target_col: str) -> None:
        self.lags = sorted(lags)
        self.target_col = target_col

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_shifted_list = []

        for lag in self.lags:
            df_shifted = df.shift(lag)
            df_shifted.columns = [f"{col}_lag{lag}" for col in df.columns]
            df_shifted_list.append(df_shifted)

        out_df = pd.concat(df_shifted_list, axis=1)
        out_df[self.target_col] = df[self.target_col]
        return out_df.dropna()

    def apply(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> DataState[None, TrainTestData, WindowingMeta]:
        extras = TrainTestData(
            train=self._create_lag_features(data.extras.train),
            test=self._create_lag_features(data.extras.test),
        )
        meta = WindowingMeta(
            n_features=len(data.extras.train.columns),
            n_lags=len(self.lags),
        )
        return DataState(None, extras, meta)
