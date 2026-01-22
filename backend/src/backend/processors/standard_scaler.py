import pandas as pd


class StandardScaler:
    def __init__(self, fit: pd.DataFrame) -> None:
        self.mean = fit.mean()
        self.std = fit.std()

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std
