import numpy as np


def symmetric_mean_absolute_percentage_error(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> np.float64:
    y_pred = y_hats[:, :, q50_idx]
    numerator = np.abs(y_trues - y_pred)
    denominator = (np.abs(y_trues) + np.abs(y_pred)) / 2
    return (numerator / np.where(denominator == 0, 1e-10, denominator)).mean() * 100
