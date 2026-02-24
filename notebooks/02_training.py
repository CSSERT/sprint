# %% Libraries
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService({"interval": "daily"})
train_loader = data_service.get("VCB")

# %% Model
lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "n_features": 5,
        "n_horizons": 3,
        "n_quantiles": 3,
    },
)

# %% Training
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=QuantileLoss([0.1, 0.5, 0.9]),
)
trainer.train(
    train_loader,
    epochs=100,
)

# %% Saving
lstm.save_pretrained(Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest")
