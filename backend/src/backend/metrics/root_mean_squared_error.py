import numpy as np

from .mean_squared_error import mean_squared_error


def root_mean_squared_error(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> np.float64:
    return np.sqrt(mean_squared_error(y_trues, y_hats, q50_idx=q50_idx))
