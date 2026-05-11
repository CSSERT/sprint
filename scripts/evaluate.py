from __future__ import annotations

import json
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
    PlottingService,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def main(
    model: str = "sprint.transformers.patchtst",
    interval: str = "daily",
) -> None:
    data_service = DataService(
        horizons=[1, 5, 10, 20],
        test_size=0.2,
        data_dir=Path.cwd() / ".." / "data" / "processed",
        feature_cols=[
            "close",
            "high",
            "low",
            "volume",
            "DIGITAL_BANKING",
            "FINANCIAL_FEE",
            "FINANCIAL_PRODUCT",
            "LEADERSHIP",
            "MACRO_REGULATION",
            "MARKET_PERCEPTION",
            "SERVICE",
        ],
    )
    _, test_loader, _ = data_service.get("VCB", interval=interval)

    model_service = ForecastingModelService.from_pretrained(
        Path.cwd() / ".." / "artifacts" / model / "latest",
    )

    metrics = MetricsEvaluatorService(
        {
            "MAE": mean_absolute_error,
            "MDA": mean_directional_accuracy,
            "RMSE": root_mean_squared_error,
            "sMAPE": symmetric_mean_absolute_percentage_error,
            "Interval Coverage": prediction_interval_coverage,
        }
    )
    y_hats, y_trues, _ = model_service.predict(test_loader)
    y_hats = data_service.inverse_y(y_hats)
    y_trues = data_service.inverse_y(y_trues)
    results = metrics.evaluate(y_trues, y_hats, q50_idx=1)

    table = Table(title=f"{model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for name, value in results.items():
        table.add_row(name, f"{value:.6f}")

    console.print(table)

    temp_dir = Path.cwd() / ".." / "temp"

    metrics_path = temp_dir / f"metrics_{model}.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    typer.echo(f"Metrics saved to '{metrics_path}'!")

    plot_path = temp_dir / f"plot_{model}.png"
    plot_service = PlottingService()
    plot_service.plot_analysis(
        data=data_service,
        model=model_service,
        ticker="VCB",
        interval=interval,
        save_path=plot_path,
    )
    typer.echo(f"Plot saved to '{plot_path}'!")


if __name__ == "__main__":
    app()
