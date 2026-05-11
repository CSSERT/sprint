from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..data.loaders import StockLoader, TickersLoader
from ..data.processors import (
    DataLoaderFactory,
    Standardizer,
    TickerEncoder,
    TrainTestSplitter,
    WindowGenerator,
)
from ..types.data import DataState, FactoryMeta
from ..types.tickers import Ticker, TickerId


@dataclass
class DataServiceProcessors:
    encoder: TickerEncoder
    splitter: TrainTestSplitter
    scaler: Standardizer
    windower: WindowGenerator
    factory: DataLoaderFactory


class DataService:
    def __init__(
        self,
        *,
        data_dir: Path | str = Path.cwd() / ".." / "data" / "raw",
        feature_cols: list[str] = ["close", "high", "low", "volume"],
        tickers_vocab_path: Path | str = (
            Path.cwd() / ".." / "data" / "tickers" / "vocab_latest.json"
        ),
        lags: list[int] = list(range(1, 31)),
        horizons: list[int],
        test_size: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        self.lags = sorted(lags)
        self.horizons = sorted(horizons)

        self.loader = StockLoader(data_dir=data_dir, feature_cols=feature_cols)
        self.tickers = TickersLoader.from_vocab(tickers_vocab_path)

        self.processors = DataServiceProcessors(
            encoder=TickerEncoder(loader=self.tickers),
            splitter=TrainTestSplitter(test_size=test_size),
            scaler=Standardizer(scale_cols=self.loader.feature_cols),
            windower=WindowGenerator(
                feature_cols=self.loader.feature_cols,
                target_col="close",
                lags=self.lags,
                horizons=self.horizons,
            ),
            factory=DataLoaderFactory(
                ticker_col=self.loader.ticker_col,
                batch_size=batch_size,
            ),
        )

    def get_raw(
        self,
        tickers: list[str] | str,
        interval: Literal["daily", "weekly"],
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        return self.loader.get_for_ticker(tickers, interval=interval)

    def get(
        self,
        tickers: list[str] | str,
        interval: Literal["daily", "weekly"],
    ) -> tuple[DataLoader, DataLoader, FactoryMeta]:
        df = self.get_raw(tickers, interval)
        data = DataState(df, extras=None, meta=None)

        data = self.processors.encoder.apply(data)
        data = self.processors.splitter.apply(data)

        self.processors.scaler.prepare(data)
        data = self.processors.scaler.apply(data)

        data = self.processors.windower.apply(data)
        data = self.processors.factory.apply(data)

        return data.extras.train, data.extras.test, data.meta

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.processors.scaler._inverse_scale(y)

    def decode_ticker(self, ticker_id: TickerId) -> Ticker:
        return self.tickers.decode(ticker_id)
