from typing import Literal

from backend.services.metrics_evaluator_service import MetricsEvaluatorService

from ..data.processors import WalkForwardSplitter
from ..types.data import DataState, TrainTestData
from .data_service import DataService
from .trainer_service import TrainerService


class CrossEvaluatorService:
    def __init__(
        self,
        *,
        trainer: TrainerService,
        data: DataService,
        metrics: MetricsEvaluatorService,
        n_folds: int = 5,
        gap: int = 5,
        test_size: int = 60,
        min_train_size: int = 200,
        expanding: bool = True,
    ) -> None:
        self.trainer = trainer
        self.data = data
        self.metrics = metrics

        q50_idx = [
            idx
            for idx, quantile in enumerate(
                self.trainer.forecaster.model.quantiles.tolist()
            )
            if round(quantile, 1) == 0.5
        ]
        if len(q50_idx) == 0:
            raise ValueError("Q50 median not found")
        self.q50_idx = q50_idx[0]

        self.splitter = WalkForwardSplitter(
            n_folds=n_folds,
            gap=gap,
            test_size=test_size,
            min_train_size=min_train_size,
            expanding=expanding,
        )

    def _process_fold(
        self,
        fold: TrainTestData,
        epochs_per_fold: int,
    ) -> dict[str, float]:
        data = DataState(None, fold, meta=None)

        self.data.processors.scaler.prepare(data)
        data = self.data.processors.scaler.apply(data)

        data = self.data.processors.windower.apply(data)
        data = self.data.processors.factory.apply(data)

        self.trainer.train(
            data.extras.train,
            epochs=epochs_per_fold,
        )

        y_hats, y_trues, _ = self.trainer.forecaster.predict(data.extras.test)
        y_hats = self.data.inverse_y(y_hats)
        y_trues = self.data.inverse_y(y_trues)

        return self.metrics.evaluate(y_trues, y_hats, q50_idx=self.q50_idx)

    def evaluate(
        self,
        tickers: list[str] | str,
        *,
        interval: Literal["daily", "weekly"],
        epochs_per_fold: int,
    ) -> list[dict[str, float]]:
        df = self.data.get_raw(tickers, interval)
        data = DataState(df, extras=None, meta=None)

        data = self.data.processors.encoder.apply(data)
        data = self.splitter.apply(data)

        return [self._process_fold(fold, epochs_per_fold) for fold in data.extras]
