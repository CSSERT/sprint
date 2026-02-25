import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData, WindowMeta

type FeatureColumns = list[str]
type TargetColumns = list[str]


class WindowGenerator(DataProcessor):
    def __init__(
        self,
        *,
        target_col: str,
        lags: list[int],
        horizons: list[int],
    ) -> None:
        self.lags = lags
        self.horizons = horizons
        self.target_col = target_col

    def _create_lag_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, FeatureColumns, TargetColumns]:
        df_shifted_list = []
        feature_columns = []
        target_columns = []

        for lag in self.lags:
            df_past = df.shift(lag)
            past_columns = [f"{col}_t-{lag}" for col in df.columns]
            df_past.columns = past_columns

            feature_columns.extend(past_columns)
            df_shifted_list.append(df_past)

        for horizon in self.horizons:
            future_column_name = f"{self.target_col}_t+{horizon}"
            df_future = (
                df.loc[:, self.target_col].shift(-horizon).rename(future_column_name)
            )

            target_columns.append(future_column_name)
            df_shifted_list.append(df_future)

        out_df = pd.concat(df_shifted_list, axis=1)
        return out_df.dropna(), feature_columns, target_columns

    def apply(
        self,
        data: DataState[None, TrainTestData, None],
    ) -> DataState[None, TrainTestData, WindowMeta]:
        train_shifted, feature_columns, target_columns = self._create_lag_features(
            data.extras.train
        )
        test_shifted, _, _ = self._create_lag_features(data.extras.test)

        extras = TrainTestData(
            train=train_shifted,
            test=test_shifted,
        )
        meta = WindowMeta(
            feature_columns=feature_columns,
            target_columns=target_columns,
            n_lags=len(self.lags),
        )
        return DataState(None, extras, meta)
