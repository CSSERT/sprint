from pathlib import Path

import typer
from backend.data.loaders import StockLoader
from backend.external import ViBankACSAPI

app = typer.Typer()


@app.command()
def main(
    ticker: str,
    interval: str = "daily",
    lookback_days: str = "month",
) -> None:
    data_dir = Path.cwd() / ".." / "data" / "raw"

    loader = StockLoader(data_dir)
    df = loader.get_for_ticker(ticker, interval=interval)

    typer.echo("Found '{ticker}' in '{data_dir}'!")
    typer.echo("Fetching sentiment data for '{ticker}'...")

    service = ViBankACSAPI()
    response = service.crawl_and_analyze(
        ticker,
        lookback_days=lookback_days,
    )


if __name__ == "__main__":
    app()
