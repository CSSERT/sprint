# %% Libraries
from backend.loaders import StockLoader
from backend.processors import StandardScaler, train_test_split

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
scaler = StandardScaler(fit=train_df)
scaled_train_df = scaler.scale(train_df)
scaled_test_df = scaler.scale(test_df)
