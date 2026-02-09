# %% Libraries
from torch.utils.data import DataLoader

from backend.datasets import WindowedDataset
from backend.loaders import StockLoader
from backend.processors import Standardizer, train_test_split
from backend.services import ForecastingModelService

# %% Load data
loader = StockLoader(
    data_dir="../data/raw",
    interval="daily",
)
df = loader.get_for_ticker("VCB")
df.head()

# %% Train test split
test_size = 0.2
train_df, test_df = train_test_split(df, test_size)
print(train_df.shape, test_df.shape)

# %% Scale
scaler = Standardizer()
scaler.fit(train_df)
scaled_train_df = scaler.transform(train_df)
scaled_test_df = scaler.transform(test_df)

# %% Dataset
feature_cols = ["volume", "close"]
target_col = "close"
window_size = 5
train_dataset = WindowedDataset(train_df, feature_cols, target_col, window_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataset = WindowedDataset(test_df, feature_cols, target_col, window_size)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% Model

lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "input_size": len(feature_cols) * window_size,
        "output_size": 1,
    },
)
