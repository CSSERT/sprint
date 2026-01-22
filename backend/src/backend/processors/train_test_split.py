import pandas as pd


def train_test_split(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples = len(df)
    split = int(n_samples * (1 - test_size))
    return df.iloc[0:split], df.iloc[split:]
