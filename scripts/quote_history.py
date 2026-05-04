from pathlib import Path
import io
import contextlib
from time import sleep
from typing import Literal
from datetime import datetime
import pandas as pd
import typer
from vnstock import Listing, Quote

app = typer.Typer()


@app.command()
def main(
    tickers: list[str] | None = None,
    ticker_group: str | None = None,
    interval: Literal["daily", "weekly"] = "daily",
    save: bool = False,
    start: datetime = "2025-10-01",
    end: datetime = "2025-12-31",
) -> None:
    if tickers is None and ticker_group is None:
        raise ValueError("Please provide `--tickers` or `--tickers-group`!")

    # `--tickers` take precedence!
    if tickers is None:
        listing = Listing()
        tickers = listing.symbols_by_group(ticker_group)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    for i, ticker in enumerate(tickers):
        typer.echo(f"Fetching for '{ticker}'...")

        quote = Quote(symbol=ticker)

        # :)
        with contextlib.redirect_stdout(io.StringIO()):
            df = quote.history(
                start=start_str,
                end=end_str,
                interval=interval[0],
            )

        print(df.head(5))

        if save:
            save_path = (
                Path.cwd()
                / ".."
                / "data"
                / "raw"
                / interval
                / f"{ticker}_{start_str}_{end_str}.csv"
            )
            df.to_csv(save_path)

            typer.echo(f"Saved to '{save_path}'!")

        # Prevents rate limiting (20req/m) :C
        if i < len(tickers) - 1:
            sleep(3)


if __name__ == "__main__":
    app()
