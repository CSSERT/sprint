import numpy as np


def mean_directional_accuracy(
    y_trues: np.ndarray,
    y_hats: np.ndarray,
    *,
    q50_idx: int,
) -> np.float64:
    trues_dir = np.sign(y_trues[1:] - y_trues[:-1])
    hats_dir = np.sign(y_hats[1:, :, q50_idx] - y_trues[:-1])
    return (trues_dir == hats_dir).mean()
