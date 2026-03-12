import numpy as np


def r2_score(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> float:
    sumsq_residuals = np.sum((y_trues - y_hats[:, :, q50_idx]) ** 2)
    sumsq_total = np.sum((y_trues - y_hats[:, :, q50_idx].mean()) ** 2)
    return 0.0 if sumsq_total == 0 else 1 - sumsq_residuals / sumsq_total
