# %% Libraries
from datetime import datetime
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(horizons=[1, 5, 10, 20], test_size=0.05)
train_loader, test_loader, meta = data_service.get(
    ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"],
    interval="daily",
)

# %% Model
model = ForecastingModelService(
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
    forecaster=model,
    optimizer=optim.AdamW(model.model.parameters(), lr=1e-3),
    criterion=QuantileLoss(model.model.quantiles.tolist()),
)
trainer.train(
    train_loader,
    epochs=100,
    save_every=100,
    save_dir=Path.cwd() / ".." / "artifacts" / "checkpoints",
    val_loader=test_loader,
    early_stopping_patience=10,
)

# %% Saving
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save_pretrained(
    Path.cwd() / ".." / "artifacts" / "sprint.transformers.patchtst" / timestamp
)
