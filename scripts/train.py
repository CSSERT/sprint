from pathlib import Path

import backend.models.bootstrap  # noqa: F401
import torch.optim as optim
import typer
from backend.losses import QuantileLoss
from backend.services import (
    DataService,
    ForecastingModelService,
    TrainerService,
)

app = typer.Typer()


@app.command()
def main(
    model: str = "sprint.transformers.patchtst",
    interval: str = "daily",
    horizons: list[int] = [1, 5, 10, 20],
    test_size: float = 0.2,
    batch_size: int = 8,
    epochs: int = 20,
    patience: int = 5,
) -> None:
    data_service = DataService(
        horizons=horizons,
        test_size=test_size,
        batch_size=batch_size,
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
    train_loader, test_loader, meta = data_service.get("VCB", interval=interval)

    _model = ForecastingModelService(
        model,
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
        forecaster=_model,
        optimizer=optim.AdamW(_model.model.parameters(), lr=1e-3),
        criterion=QuantileLoss(_model.model.quantiles.tolist()),
    )

    typer.echo(f"Training '{model}' for {epochs} epochs...")

    trainer.train(
        train_loader,
        epochs=epochs,
        val_loader=test_loader,
        early_stopping_patience=patience,
    )

    save_path = Path.cwd() / ".." / "artifacts" / model / "latest"
    _model.save_pretrained(save_path)

    typer.echo(f"Saved to '{save_path}'!")


if __name__ == "__main__":
    app()
