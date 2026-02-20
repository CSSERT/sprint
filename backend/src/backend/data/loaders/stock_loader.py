from pathlib import Path

import pandas as pd


class StockLoader:
    def __init__(self, data_dir: Path | str, interval: str) -> None:
        self.data_dir = Path(data_dir)
        self.interval = interval

    def get_for_ticker(self, ticker: str) -> pd.DataFrame:
        df = pd.read_csv(
            self.data_dir / self.interval / f"{ticker}.csv",
            parse_dates=True,
            index_col="time",
        )
        df = df.drop(["Unnamed: 0"], axis=1)
        return df
