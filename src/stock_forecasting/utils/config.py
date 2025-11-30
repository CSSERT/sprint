from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Cfg:
  # Common
  SEED = 420
  
  # Paths
  DATA_DIR = Path.cwd() / ".." / "data"

  # Dataset
  INTERVAL = "day"
  TICKERS = ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"]

  SLIDING_WINDOW_SIZE = 4
  
  FEATURE_COLS = ["close", "volume", "volatility"]
  TARGET_COL = "close"

  VAL_SIZE_RATIO = 0.05
  TEST_SIZE_RATIO = 0.05

  BATCH_SIZE = 32

  # Training
  NUM_EPOCHS = 200