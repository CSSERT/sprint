import json
from pathlib import Path
from typing import Literal

from ...types.tickers import Ticker, TickerId, TickersVocab


class TickersLoader:
    def __init__(self, vocab: TickersVocab | None = None) -> None:
        self.vocab = vocab or {}

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        *,
        interval: Literal["daily", "weekly"],
        file_exts: list[str] = [".csv"],
    ) -> "TickersLoader":
        tickers_loader = cls()
        for file in (Path(data_dir) / interval).iterdir():
            if file.suffix not in file_exts:
                continue
            tickers_loader.encode(file.stem)
        return tickers_loader

    @classmethod
    def from_vocab(cls, vocab_path: str | Path) -> "TickersLoader":
        with open(vocab_path) as vocab_file:
            vocab = json.load(vocab_file)
        return cls(vocab)

    def encode(self, ticker: Ticker) -> TickerId:
        if ticker not in self.vocab:
            self.vocab[ticker] = len(self.vocab)
        return self.vocab[ticker]

    def decode(self, ticker_id: TickerId) -> Ticker:
        for ticker, _ticker_id in self.vocab.items():
            if _ticker_id == ticker_id:
                return ticker
        raise KeyError(ticker_id)

    def save(self, vocab_path: str | Path) -> None:
        with open(vocab_path, "w") as vocab_file:
            json.dump(self.vocab, vocab_file)
