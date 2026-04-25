import numpy as np


def mean_squared_error(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> np.float64:
    return ((y_trues - y_hats[:, :, q50_idx]) ** 2).mean()
