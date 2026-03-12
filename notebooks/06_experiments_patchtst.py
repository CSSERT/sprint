# %% Libraries
import backend.models.bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(
    lags=list(range(1, 31)),
    horizons=list(range(1, 8)),
    test_size=0.2,
)
train_loader, test_loader, meta = data_service.get("VCB", interval="daily")

# %% Model
patchtst = ForecastingModelService(
    "sprint.transformers.patchtst",
    {
        "n_features": len(data_service.loader.feature_cols),
        "n_tickers": len(data_service.tickers.vocab),
        "n_lags": len(data_service.lags),
        "horizons": data_service.horizons,
        "quantiles": [0.1, 0.5, 0.9],
        "feature_target_idx": meta.feature_target_idx,
    },
)

# %% Training
trainer = TrainerService(
    forecaster=patchtst,
    optimizer=optim.AdamW(patchtst.model.parameters(), lr=1e-3),
    criterion=QuantileLoss(patchtst.model.quantiles.tolist()),
)
trainer.train(
    train_loader,
    epochs=1,
)

# %% Plotting latest
y_hats, y_trues, tickers = patchtst.predict(test_loader)

y_hats = data_service.inverse_y(y_hats)
y_trues = data_service.inverse_y(y_trues)

ticker_id = data_service.tickers.encode("VCB")
ticker_indicies = [i for i, ticker in enumerate(tickers) if ticker == ticker_id]

y_hat = y_hats[ticker_indicies[-1]]
y_true = y_trues[ticker_indicies[-1]]

H = y_hat.shape[0]
x = range(1, H + 1)

plt.plot(x, y_true, color="black", label="True")
plt.plot(x, y_hat[:, 1], "--", label="Median Prediction")
plt.fill_between(x, y_hat[:, 0], y_hat[:, 2], alpha=0.2)

plt.legend()
plt.show()

# %% Plotting
y_hats, y_trues, tickers = patchtst.predict(test_loader)

y_hats = data_service.inverse_y(y_hats)
y_trues = data_service.inverse_y(y_trues)

ticker_id = data_service.tickers.encode("VCB")
ticker_indicies = [i for i, ticker in enumerate(tickers) if ticker == ticker_id]

plt.figure()

for i, idx in enumerate(ticker_indicies[-5:]):
    y_hat = y_hats[idx]
    y_true = y_trues[idx]

    H = y_hat.shape[0]
    x = range(i * H, i * H + H)

    plt.plot(x, y_true, color="black")
    plt.plot(x, y_hat[:, 1], "--")
    plt.fill_between(x, y_hat[:, 0], y_hat[:, 2], alpha=0.2)

# plt.legend()
plt.show()
