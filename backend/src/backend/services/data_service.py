from pathlib import Path
from typing import NotRequired, Required, TypedDict, cast

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


class DataServiceConfig(TypedDict):
    data_dir: NotRequired[str | Path]
    interval: Required[str]

    test_size: NotRequired[float]
    lags: NotRequired[list[int]]
    horizons: NotRequired[list[int]]


class DataService:
    def __init__(
        self,
        config: DataServiceConfig,
    ) -> None:
        self.loader = StockLoader(
            data_dir=config.get("data_dir", Path.cwd() / ".." / "data" / "raw"),
            interval=config["interval"],
        )
        self.processors = [
            TrainTestSplitter(test_size=config.get("test_size", 0.0)),
            Standardizer(),
            WindowGenerator(
                target_col="close",
                lags=config.get("lags", [1, 7, 30]),
                horizons=config.get("horizons", [1, 3, 7]),
            ),
            DataLoaderFactory(),
        ]
        self.should_include_test_set = config.get("test_size", 0.0) > 0.0

    def get(self, ticker: str) -> DataLoader | tuple[DataLoader, DataLoader]:
        df = self.loader.get_for_ticker(ticker)
        data = DataState(df, extras=None, meta=None)

        for processor in self.processors:
            processor.prepare(data)
            data = processor.apply(data)

        data = cast(DataState[None, TrainTestLoader, None], data)
        if self.should_include_test_set:
            return data.extras.train, data.extras.test
        return data.extras.train

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.processors[1]._inverse_scale(y)
