# %% Libraries
import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import (
    DataService,
    ForecastingModelService,
    PlottingService,
    TrainerService,
)

# %% Load data
data_service = DataService(
    lags=list(range(1, 31)),
    horizons=list(range(1, 8)),
    test_size=0.2,
)
train_loader, test_loader, _ = data_service.get("VCB", interval="daily")

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
    epochs=20,
)

# %% Plotting
plot_service = PlottingService()
plot_service.plot_analysis(
    data=data_service,
    model=lstm,
    ticker="VCB",
    interval="daily",
)
