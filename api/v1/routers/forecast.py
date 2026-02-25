from typing import cast

from backend.services import DataService, ForecastingModelService
from fastapi import APIRouter, Depends, Request
from torch.utils.data import DataLoader

from ..schemas import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


def get_data(req: Request) -> DataService:
    return req.app.state.data


def get_model(req: Request) -> ForecastingModelService:
    return req.app.state.model


@router.post("/", response_model=ForecastResponse)
def forecast(
    req: ForecastRequest,
    data: DataService = Depends(get_data),
    model: ForecastingModelService = Depends(get_model),
) -> ForecastResponse:
    loader = cast(DataLoader, data.get(req.ticker))

    predictions = model.predict(loader)
    prediction_last = predictions[-1]

    horizons = data.horizons
    quantiles = model.model.quantiles.tolist()

    return ForecastResponse(
        predictions=[
            {
                "step": horizon,
                "quantiles": {
                    f"{quantile:.1f}": round(float(prediction_last[i, j]), 4)
                    for j, quantile in enumerate(quantiles)
                },
            }
            for i, horizon in enumerate(horizons)
        ]
    )
