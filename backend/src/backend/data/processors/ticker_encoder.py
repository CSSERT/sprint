import pandas as pd

from backend.data.loaders import TickersLoader

from ...types.data import DataProcessor, DataState


class TickerEncoder(DataProcessor):
    def __init__(self, *, loader: TickersLoader) -> None:
        self.loader = loader

    def apply(
        self,
        data: DataState[pd.DataFrame, None, None],
    ) -> DataState[pd.DataFrame, None, None]:
        data._default["ticker"] = data._default["ticker"].map(self.loader.encode)
        return data
