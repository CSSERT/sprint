import pandas as pd

from ...types.data import DataProcessor, DataState, TrainTestData, WindowMeta

type FeatureColumns = list[str]
type TargetColumns = list[str]


class WindowGenerator(DataProcessor):
    def __init__(
        self,
        *,
        feature_cols: list[str],
        target_col: str,
        lags: list[int],
        horizons: list[int],
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lags = lags
        self.horizons = horizons

    def _create_lag_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, FeatureColumns, TargetColumns]:
        features_df = df.loc[:, self.feature_cols]
        other_cols = [col for col in df.columns if col not in self.feature_cols]

        df_shifted_list = []
        feature_columns = []
        target_columns = []

        for lag in self.lags:
            df_past = features_df.shift(lag)
            past_columns = [f"{col}_t-{lag}" for col in features_df.columns]
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

        df_shifted_list.append(df.loc[:, other_cols])
        return (
            pd.concat(df_shifted_list, axis=1).dropna(),
            feature_columns,
            target_columns,
        )

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
            feature_target_idx=self.feature_cols.index(self.target_col),
        )
        return DataState(None, extras, meta)
