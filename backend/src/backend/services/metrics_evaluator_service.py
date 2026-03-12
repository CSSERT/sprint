import numpy as np

from ..metrics import Metric


class MetricsEvaluatorService:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        self.metrics = metrics

    def evaluate(
        self,
        y_trues: np.ndarray,
        y_hats: np.ndarray,
        *,
        q50_idx: int,
    ) -> dict[str, float]:
        return {
            name: metric(y_trues, y_hats, q50_idx=q50_idx)
            for name, metric in self.metrics.items()
        }
