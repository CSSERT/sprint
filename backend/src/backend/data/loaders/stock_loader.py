from pathlib import Path

import pandas as pd


class StockLoader:
    def __init__(
        self,
        data_dir: Path | str,
        interval: str,
        keep_columns: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.keep_columns = keep_columns or []

    def get_for_ticker(self, ticker: str) -> pd.DataFrame:
        df = pd.read_csv(
            self.data_dir / self.interval / f"{ticker}.csv",
            parse_dates=True,
            index_col="time",
        )
        df = df.drop(["Unnamed: 0"], axis=1)
        if len(self.keep_columns) >= 1:
            df = df.loc[:, self.keep_columns]
        return df
