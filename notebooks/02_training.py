# %% Libraries
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(horizons=[1, 5, 10, 20], test_size=0.2)
train_loader, test_loader, meta = data_service.get(
    ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"],
    interval="daily",
)

# %% Models to train
model_names = [
    "sprint.linear.dlinear",
    "sprint.rnn.lstm",
    "sprint.rnn.patchlstm",
    "sprint.transformers.vanilla",
    "sprint.transformers.patchtst",
    "sprint.transformers.nst",
]

# %% Training loop
for model_name in model_names:
    print(f"Training: '{model_name}'...\n{'=' * 40}")

    model = ForecastingModelService(
        model_name,
        {
            "n_features": len(data_service.loader.feature_cols),
            "n_tickers": len(data_service.tickers.vocab),
            "n_lags": len(data_service.lags),
            "horizons": data_service.horizons,
            "quantiles": [0.1, 0.5, 0.9],
            "feature_target_idx": meta.feature_target_idx,
        },
    )

    trainer = TrainerService(
        forecaster=model,
        optimizer=optim.AdamW(model.model.parameters(), lr=1e-3),
        criterion=QuantileLoss(model.model.quantiles.tolist()),
    )
    trainer.train(
        train_loader,
        epochs=20,
        val_loader=test_loader,
        early_stopping_patience=5,
    )

    # %% Saving
    save_path = Path.cwd() / ".." / "artifacts" / model_name / "latest"
    model.save_pretrained(save_path)
    print(f"Saved: {save_path}")
