from typing import Protocol

import numpy as np

from .mean_absolute_error import mean_absolute_error
from .mean_directional_accuracy import mean_directional_accuracy
from .r2_score import r2_score


class Metric(Protocol):
    def __call__(
        self,
        y_trues: np.ndarray,
        y_hats: np.ndarray,
        *,
        q50_idx: int,
    ) -> float: ...


__all__ = ["mean_directional_accuracy", "mean_absolute_error", "r2_score"]
