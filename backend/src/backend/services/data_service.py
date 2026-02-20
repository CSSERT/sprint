from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Required, TypedDict, cast

from torch.utils.data import DataLoader

from backend.types.data import DataState

from ..data.loaders import StockLoader
from ..data.processors import Standardizer, TrainTestSplitter, WindowGenerator


@dataclass
class TrainTestLoader:
    train: DataLoader
    test: DataLoader


class DataServiceConfig(TypedDict):
    data_dir: Required[str | Path]
    interval: Required[str]
    feature_cols: Required[list[str]]
    target_col: Required[str]

    test_size: NotRequired[float]
    window_size: NotRequired[int]
    batch_size: NotRequired[int]


class DataService:
    def __init__(
        self,
        config: DataServiceConfig,
    ) -> None:
        self.loader = StockLoader(
            data_dir=config["data_dir"],
            interval=config["interval"],
        )
        self.processors = [
            TrainTestSplitter(test_size=config.get("test_size", 0.2)),
            Standardizer(),
            WindowGenerator(
                feature_cols=config["feature_cols"],
                target_col=config["target_col"],
                window_size=config.get("window_size", 5),
                batch_size=config.get("batch_size", 32),
            ),
        ]

    def get(self, ticker: str) -> tuple[DataLoader, DataLoader]:
        df = self.loader.get_for_ticker(ticker)
        data = DataState(df, None)

        for processor in self.processors:
            processor.prepare(data)
            data = processor.apply(data)

        data = cast(DataState[None, TrainTestLoader], data)
        return data.extras.train, data.extras.test
