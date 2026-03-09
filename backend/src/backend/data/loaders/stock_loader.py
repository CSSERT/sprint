from pathlib import Path
from typing import Literal

import pandas as pd


class StockLoader:
    def __init__(
        self,
        data_dir: Path | str,
        *,
        feature_cols: list[str] = ["close", "high", "low", "volume"],
        time_col: str = "time",
        ticker_col: str = "ticker",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.ticker_col = ticker_col

    def get_for_ticker(
        self,
        tickers: list[str] | str,
        *,
        interval: Literal["daily", "weekly"],
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]

        full_data_path = self.data_dir / interval
        dfs: list[pd.DataFrame] = []

        for ticker in tickers:
            df = pd.read_csv(
                full_data_path / f"{ticker}.csv",
                usecols=self.feature_cols + [self.time_col],
                parse_dates=[self.time_col],
            )
            dfs.append(df.assign(**{self.ticker_col: ticker}))

        out_df = pd.concat(dfs, ignore_index=True)
        out_df[self.ticker_col] = out_df[self.ticker_col].astype("category")

        return out_df.sort_values([self.ticker_col, self.time_col]).set_index(
            self.time_col
        )
