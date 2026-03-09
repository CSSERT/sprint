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
train_loader, test_loader = data_service.get("VCB", interval="daily")

# %% Model
lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "n_features": len(data_service.loader.feature_cols),
        "n_tickers": len(data_service.tickers.vocab),
        "horizons": data_service.horizons,
        "quantiles": [0.1, 0.5, 0.9],
    },
)

# %% Training
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=QuantileLoss(lstm.model.quantiles.tolist()),
)
trainer.train(
    train_loader,
    epochs=10,
)

# %% Plotting latest
y_hats, y_trues, _ = lstm.predict(test_loader)
y_hats = data_service.inverse_y(y_hats)
y_trues = data_service.inverse_y(y_trues)

y_hat_latest = y_hats[-1]
y_true_latest = y_trues[-1]

x = range(1, y_hat_latest.shape[0] + 1)

plt.plot(x, y_true_latest, color="black", label="True")
plt.plot(x, y_hat_latest[:, 1], "--", label="Median Prediction")
plt.fill_between(x, y_hat_latest[:, 0], y_hat_latest[:, 2], alpha=0.2)

plt.legend()
plt.show()
