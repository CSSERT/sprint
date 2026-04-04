# %% Libraries
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(horizons=[1, 5, 10, 20])
train_loader, test_loader, _ = data_service.get(
    ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"],
    interval="daily",
)

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
    epochs=1,
)

# %% Saving
lstm.save_pretrained(Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest")
