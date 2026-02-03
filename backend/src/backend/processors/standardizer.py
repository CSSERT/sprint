import pandas as pd


class Standardizer:
    def fit(self, df: pd.DataFrame) -> None:
        self.mean = df.mean()
        self.std = df.std()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std
