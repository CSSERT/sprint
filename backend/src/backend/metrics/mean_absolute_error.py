import numpy as np


def mean_absolute_error(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> float:
    return np.abs(y_trues - y_hats[:, :, q50_idx]).mean()
