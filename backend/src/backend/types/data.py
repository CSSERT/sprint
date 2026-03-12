from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd
from torch.utils.data import DataLoader

TDefault = TypeVar("TDefault")
TExtras = TypeVar("TExtras")
TMeta = TypeVar("TMeta")


class DataState(Generic[TDefault, TExtras, TMeta]):
    def __init__(self, _default: TDefault, extras: TExtras, meta: TMeta) -> None:
        self._default = _default
        self.extras = extras
        self.meta = meta


@dataclass
class TrainTestData:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TrainTestLoader:
    train: DataLoader
    test: DataLoader


@dataclass
class WindowMeta:
    feature_columns: list[str]
    target_columns: list[str]
    n_lags: int
    feature_target_idx: int


@dataclass
class FactoryMeta:
    feature_target_idx: int


class DataProcessor:
    def prepare(self, data: DataState) -> None: ...

    def apply(self, data: DataState) -> DataState: ...
