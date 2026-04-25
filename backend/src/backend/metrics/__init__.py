from typing import Protocol

import numpy as np

from .mean_absolute_error import mean_absolute_error
from .mean_directional_accuracy import mean_directional_accuracy
from .mean_squared_error import mean_squared_error
from .prediction_interval_coverage import (
    prediction_interval_coverage,
)
from .root_mean_squared_error import root_mean_squared_error
from .symmetric_mean_absolute_percentage_error import (
    symmetric_mean_absolute_percentage_error,
)


class Metric(Protocol):
    def __call__(
        self,
        y_trues: np.ndarray,
        y_hats: np.ndarray,
        *,
        q50_idx: int,
    ) -> float: ...


__all__ = [
    "mean_directional_accuracy",
    "mean_absolute_error",
    "prediction_interval_coverage",
    "mean_squared_error",
    "root_mean_squared_error",
    "symmetric_mean_absolute_percentage_error",
]
