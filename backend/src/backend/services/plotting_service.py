from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..types.data import DataState
from .data_service import DataService
from .forecasting_model_service import ForecastingModelService


class ForecastAnalysisBundle(TypedDict):
    predictions: np.ndarray
    test_timestamps: pd.DatetimeIndex
    horizons: list[int]
    quantiles: list[float]


class PlottingService:
    def plot_analysis(
        self,
        predictions: np.ndarray | None = None,
        *,
        data: DataService,
        model: ForecastingModelService | None = None,
        ticker: str,
        interval: str,
        lookback_days: int = 150,
        save_path: str | Path | None = None,
    ) -> None:
        if predictions is None:
            if model is None:
                raise ValueError("model required when predictions not provided")
            _, loader, _ = data.get(ticker, interval=interval)
            predictions, _, _ = model.predict(loader)
            predictions = data.inverse_y(predictions)

        horizons = data.horizons
        quantiles = model.model.quantiles.tolist()

        test_timestamps = self._get_test_timestamps(data, ticker, interval)

        bundle: ForecastAnalysisBundle = {
            "predictions": predictions,
            "test_timestamps": test_timestamps,
            "horizons": horizons,
            "quantiles": quantiles,
        }

        plot_df = data.get_raw(ticker, interval).tail(lookback_days)

        n_horizons = len(horizons)
        rows, cols = (n_horizons + 1) // 2, 2
        _, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True)
        axes = axes.flatten()

        for horizon_idx in range(n_horizons):
            self._plot_horizon(
                axes[horizon_idx],
                horizon_idx,
                horizons[horizon_idx],
                n_horizons,
                bundle,
                plot_df,
            )

        for j in range(n_horizons, len(axes)):
            plt.delaxes(axes[j])

        plt.suptitle(f"Forecast Analysis: {ticker}", y=1.02)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _get_test_timestamps(
        self,
        data: DataService,
        ticker: str,
        interval: str,
    ) -> pd.DatetimeIndex:
        full_df = data.get_raw(ticker, interval)
        state = DataState(full_df, extras=None, meta=None)
        state = data.processors.splitter.apply(state)
        test_df = state.extras.test

        lags_offset = max(data.lags)
        horizons_offset = max(data.horizons)
        return test_df.index[lags_offset:-horizons_offset]

    def _plot_horizon(
        self,
        ax: plt.Axes,
        horizon_idx: int,
        horizon: int,
        n_horizons: int,
        bundle: ForecastAnalysisBundle,
        plot_df: pd.DataFrame,
    ) -> None:
        predictions = bundle["predictions"]
        test_timestamps = bundle["test_timestamps"]
        quantiles = bundle["quantiles"]

        median_idx = len(quantiles) // 2
        x_indices, y_median, y_low, y_high = [], [], [], []

        for i, t in enumerate(test_timestamps):
            try:
                pos = plot_df.index.get_loc(t)
                target_date = plot_df.index[pos + horizon]
                if target_date in plot_df.index:
                    x_indices.append(plot_df.index.get_loc(target_date))
                    y_median.append(predictions[i, horizon_idx, median_idx])
                    y_low.append(predictions[i, horizon_idx, 0])
                    y_high.append(predictions[i, horizon_idx, -1])
            except IndexError, KeyError:
                continue

        ax.plot(np.arange(len(plot_df)), plot_df["close"], alpha=0.3, label="Actual")

        if x_indices:
            color = plt.get_cmap("plasma")(horizon_idx / n_horizons)
            ax.plot(x_indices, y_median, color=color, label=f"H+{horizon}")
            ax.fill_between(
                x_indices, y_low, y_high, color=color, alpha=0.15, label="CI"
            )

        ax.set_title(f"Horizon +{horizon}")
        self._format_axis(ax, plot_df)
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.6)

    def _format_axis(self, ax: plt.Axes, plot_df: pd.DataFrame) -> None:
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        def format_date(x: float, _: int | None = None) -> str:
            idx = int(round(x))
            if 0 <= idx < len(plot_df):
                return plot_df.index[idx].strftime("%Y-%m")
            return ""

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
