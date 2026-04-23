from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data_service import DataService
from .forecasting_model_service import ForecastingModelService


class PlottingService:
    def __init__(self, style: str = "seaborn-v0_8-muted") -> None:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("ggplot")

    def plot_analysis(
        self,
        *,
        data: DataService,
        model: ForecastingModelService,
        ticker: str,
        interval: Literal["daily", "weekly"],
        lookback_days: int = 150,
    ) -> None:
        data_bundle = self._prepare_analysis_data(data, model, ticker, interval)

        n_horizons = len(data_bundle["horizons"])
        cols = 2
        rows = (n_horizons + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True)
        axes = axes.flatten()

        plot_df = data_bundle["full_df"].tail(lookback_days)

        for horizon_idx in range(n_horizons):
            self._plot_horizon_subplot(
                ax=axes[horizon_idx],
                horizon_idx=horizon_idx,
                horizon_value=data_bundle["horizons"][horizon_idx],
                n_horizons=n_horizons,
                data_bundle=data_bundle,
                plot_df=plot_df,
            )

        for j in range(n_horizons, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"Continuous Horizon-Wise Forecast Analysis: {ticker}",
            fontsize=22,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def _prepare_analysis_data(
        self,
        data: DataService,
        model: ForecastingModelService,
        ticker: str,
        interval: Literal["daily", "weekly"],
    ) -> dict:
        _, loader, _ = data.get(ticker, interval=interval)

        predictions, _, tickers = model.predict(loader)
        predictions = data.inverse_y(predictions)

        ticker_id = data.tickers.encode(ticker)
        ticker_mask = np.array(tickers) == ticker_id

        full_df = data.get_raw(ticker, interval=interval)
        split_idx = int(len(full_df) * (1 - data.processors.splitter.test_size))
        test_df = full_df.iloc[split_idx:]

        lags_offset = max(data.lags)
        horizons_offset = max(data.horizons)
        test_timestamps = test_df.index[lags_offset:-horizons_offset]

        return {
            "predictions": predictions[ticker_mask],
            "test_timestamps": test_timestamps[: len(predictions[ticker_mask])],
            "full_df": full_df,
            "horizons": data.horizons,
            "quantiles": model.model.quantiles.tolist(),
        }

    def _plot_horizon_subplot(
        self,
        ax: plt.Axes,
        horizon_idx: int,
        horizon_value: int,
        n_horizons: int,
        data_bundle: dict,
        plot_df: pd.DataFrame,
    ) -> None:
        median_idx = len(data_bundle["quantiles"]) // 2
        predictions = data_bundle["predictions"]
        test_timestamps = data_bundle["test_timestamps"]
        full_df = data_bundle["full_df"]

        ax.plot(
            np.arange(len(plot_df)),
            plot_df["close"],
            color="#2d3436",
            alpha=0.3,
            label="Actual",
        )

        x_idxs, y_median, y_low, y_high = [], [], [], []

        for i in range(len(test_timestamps)):
            t = test_timestamps[i]
            try:
                full_pos = full_df.index.get_loc(t)
                target_date = full_df.index[full_pos + horizon_value]

                if target_date in plot_df.index:
                    x_idxs.append(plot_df.index.get_loc(target_date))
                    y_median.append(predictions[i, horizon_idx, median_idx])
                    y_low.append(predictions[i, horizon_idx, 0])
                    y_high.append(predictions[i, horizon_idx, -1])
            except IndexError, KeyError:
                continue

        if x_idxs:
            color = plt.get_cmap("plasma")(horizon_idx / n_horizons)
            ax.plot(
                x_idxs,
                y_median,
                color=color,
                linewidth=2,
                label=f"H+{horizon_value} Pred",
            )
            ax.fill_between(
                x_idxs, y_low, y_high, color=color, alpha=0.15, label="90% CI"
            )

        ax.set_title(
            f"Horizon {horizon_value} Prediction (+{horizon_value} Trading Days)",
            fontsize=13,
            fontweight="bold",
        )
        self._apply_aesthetics(ax, plot_df)
        ax.legend(loc="upper left", fontsize="small")

    def _apply_aesthetics(self, ax: plt.Axes, plot_df: pd.DataFrame) -> None:
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle=":", alpha=0.6)

        def format_date(x: float, pos: int | None = None) -> str:
            idx = int(round(x))
            if 0 <= idx < len(plot_df):
                return plot_df.index[idx].strftime("%Y-%m")
            return ""

        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(format_date))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
