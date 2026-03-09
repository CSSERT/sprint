# %% Libraries
from pathlib import Path

from backend.data.loaders import TickersLoader

# %% Loader
tickers = TickersLoader.from_data_dir(
    Path.cwd() / ".." / "data" / "raw",
    interval="daily",
)

# %% Save vocab
tickers.save(Path.cwd() / ".." / "data" / "tickers" / "vocab_latest.json")
