# %% Libraries
import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.metrics import mean_absolute_error, mean_directional_accuracy
from backend.services import (
    CrossEvaluatorService,
    DataService,
    ForecastingModelService,
    MetricsEvaluatorService,
    TrainerService,
)

# %% Load data
data_service = DataService(
    lags=list(range(1, 31)),
    horizons=list(range(1, 8)),
    test_size=0.2,
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

# %% Trainer
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=QuantileLoss(lstm.model.quantiles.tolist()),
)

# %% Cross-evaluation
metrics = MetricsEvaluatorService(
    {
        "MAE": mean_absolute_error,
        "MDA": mean_directional_accuracy,
    }
)
evaluator = CrossEvaluatorService(
    trainer=trainer,
    data=data_service,
    metrics=metrics,
)
results = evaluator.evaluate(["VCB", "OCB"], interval="daily", epochs_per_fold=1)
for i, result in enumerate(results):
    print(f"[Fold {i + 1}/{len(results)}] {result}")
