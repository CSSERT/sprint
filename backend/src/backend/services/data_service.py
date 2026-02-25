from pathlib import Path
from typing import Literal, cast

import numpy as np
from torch.utils.data import DataLoader

from ..data.loaders import StockLoader
from ..data.processors import (
    DataLoaderFactory,
    Standardizer,
    TrainTestSplitter,
    WindowGenerator,
)
from ..types.data import DataState, TrainTestLoader


class DataService:
    def __init__(
        self,
        *,
        data_dir: Path | str = Path.cwd() / ".." / "data" / "raw",
        interval: Literal["daily", "weekly"],
        lags: list[int] = list(range(1, 31)),
        horizons: list[int],
        test_size: float = 0.0,
        keep_columns: list[str] = ["close", "high", "low", "volume"],
    ) -> None:
        self.should_include_test_set = test_size > 0.0

        self.lags = sorted(lags)
        self.horizons = sorted(horizons)
        self.n_features = len(keep_columns)

        self.loader = StockLoader(
            data_dir=data_dir,
            interval=interval,
            keep_columns=keep_columns,
        )

        self.processors = {
            "splitting": TrainTestSplitter(
                test_size=test_size,
            ),
            "scaling": Standardizer(),
            "windowing": WindowGenerator(
                target_col="close",
                lags=self.lags,
                horizons=self.horizons,
            ),
            "final": DataLoaderFactory(),
        }

    def get(self, ticker: str) -> tuple[DataLoader, DataLoader] | DataLoader:
        df = self.loader.get_for_ticker(ticker)
        data = DataState(df, extras=None, meta=None)

        for processor in self.processors.values():
            processor.prepare(data)
            data = processor.apply(data)

        data = cast(DataState[None, TrainTestLoader, None], data)
        if self.should_include_test_set:
            return data.extras.train, data.extras.test
        return data.extras.train

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.processors["scaling"]._inverse_scale(y)
