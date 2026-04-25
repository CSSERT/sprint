import numpy as np


def prediction_interval_coverage(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> np.float64:
    lower_bound = y_hats[:, :, 0]
    upper_bound = y_hats[:, :, -1]
    return ((y_trues >= lower_bound) & (y_trues <= upper_bound)).mean()
