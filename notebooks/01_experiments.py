# %%
from backend.loaders import StockLoader

loader = StockLoader(
    data_dir="../data/raw",
    interval="daily",
)

df = loader.get_for_ticker("VCB")
df.head()
