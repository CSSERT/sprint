# %% Libraries
import json
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
from backend.metrics import (
    mean_absolute_error,
    mean_directional_accuracy,
    prediction_interval_coverage,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from backend.services import (
    DataService,
    ForecastingModelService,
    MetricsEvaluatorService,
    PlottingService,
)

# %% Load data
data_service = DataService(horizons=[1, 5, 10, 20], test_size=0.2)
_, test_loader, _ = data_service.get(
    ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"],
    interval="daily",
)

# %% Model
model = ForecastingModelService.from_pretrained(
    Path.cwd()
    / ".."
    / "artifacts"
    / "sprint.transformers.patchtst"
    / "2026-04-27_epochs-100",
)

# %% Evaluation
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

metrics = MetricsEvaluatorService(
    {
        "MAE": mean_absolute_error,
        "MDA": mean_directional_accuracy,
        "RMSE": root_mean_squared_error,
        "sMAPE": symmetric_mean_absolute_percentage_error,
        "Interval Coverage": prediction_interval_coverage,
    }
)
y_hats, y_trues, _ = model.predict(test_loader)
y_hats = data_service.inverse_y(y_hats)
y_trues = data_service.inverse_y(y_trues)
results = metrics.evaluate(y_trues, y_hats, q50_idx=1)
print(results)

metrics_path = temp_dir / "evaluation_metrics.json"
metrics_path.write_text(json.dumps(results, indent=2))
print(f"Metrics saved to {metrics_path}")

# %% Plotting
plot_service = PlottingService()
plot_service.plot_analysis(
    data=data_service,
    model=model,
    ticker="VHM",
    interval="daily",
    save_path=temp_dir / "evaluation_plot.png",
)
print(f"Plot saved to {temp_dir / 'evaluation_plot.png'}")
