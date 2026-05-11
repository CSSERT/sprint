from __future__ import annotations

# import json
from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import typer
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
    # PlottingService,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = "sprint.transformers.patchtst",
    interval: str = "daily",
) -> None:
    data_service = DataService(horizons=[1, 5, 10, 20], test_size=0.2)
    _, test_loader, _ = data_service.get(
        ["DGW", "FRT", "HPG", "NKG", "OCB", "PDR", "VCB", "VHM"],
        interval=interval,
    )

    model_path = Path.cwd() / ".." / "artifacts" / model_name / "latest"
    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    model = ForecastingModelService.from_pretrained(model_path)

    # temp = Path(temp_dir)
    # temp.mkdir(exist_ok=True)

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

    table = Table(title=f"{model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for name, value in results.items():
        table.add_row(name, f"{value:.6f}")

    console.print(table)

    # metrics_path = temp / "evaluation_metrics.json"
    # metrics_path.write_text(json.dumps(results, indent=2))
    # typer.echo(f"Metrics saved to {metrics_path}")

    # plot_service = PlottingService()
    # plot_service.plot_analysis(
    #     data=data_service,
    #     model=model,
    #     ticker=ticker,
    #     interval=interval,
    #     save_path=temp / "evaluation_plot.png",
    # )
    # typer.echo(f"Plot saved to {temp / 'evaluation_plot.png'}")


if __name__ == "__main__":
    app()
