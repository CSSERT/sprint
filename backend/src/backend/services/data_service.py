from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Required, TypedDict, cast

from torch.utils.data import DataLoader

from backend.types.data import DataState

from ..data.loaders import StockLoader
from ..data.processors import (
    DataLoaderFactory,
    Standardizer,
    TrainTestSplitter,
    WindowGenerator,
)


@dataclass
class TrainTestLoader:
    train: DataLoader
    test: DataLoader


class DataServiceConfig(TypedDict):
    data_dir: NotRequired[str | Path]
    interval: Required[str]

    test_size: NotRequired[float]
    lags: NotRequired[list[int]]


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
                lags=config.get("window_size", [1, 7, 30]),
            ),
            DataLoaderFactory(target_col="close"),
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
