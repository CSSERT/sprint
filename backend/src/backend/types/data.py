from typing import Generic, TypeVar

TDefault = TypeVar("TDefault")
TExtras = TypeVar("TExtras")


class DataState(Generic[TDefault, TExtras]):
    def __init__(self, _default: TDefault, extras: TExtras) -> None:
        self._default = _default
        self.extras = extras


class DataProcessor:
    def prepare(self, data: DataState, *args, **kwargs) -> None: ...

    def apply(self, data: DataState, *args, **kwargs) -> DataState: ...
