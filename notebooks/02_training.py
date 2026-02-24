# %% Libraries
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.nn as nn
import torch.optim as optim
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(
    {
        "interval": "daily",
        "test_size": 0.2,
    }
)
train_loader, test_loader = data_service.get("VCB")

# %% Model
lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "input_size": 5,
        "output_size": 1,
    },
)

# %% Training
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=nn.MSELoss(),
)
trainer.train(
    train_loader,
    epochs=100,
)

# %% Saving
lstm.save_pretrained(Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest")
