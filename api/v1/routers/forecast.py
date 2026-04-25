import pandas as pd
from backend.services import DataService, ForecastingModelService
from fastapi import APIRouter, Depends, HTTPException, Request

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
    try:
        loader, _, _ = data.get(req.ticker, req.interval)
        raw_df = data.get_raw(req.ticker, req.interval)
    except FileNotFoundError, KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{req.ticker}' not found for {req.interval} interval",
        )

    last_date = raw_df.index[-1]

    history = None
    if req.lookback_days is not None:
        history = (
            raw_df.tail(req.lookback_days)
            .assign(date=lambda x: x.index.strftime("%Y-%m-%d"))
            .loc[:, ["date", "close"]]
            .to_dict(orient="records")
        )

    predictions, _, _ = model.predict(loader)
    prediction_last = data.inverse_y(predictions[-1])

    horizons = data.horizons
    quantiles = model.model.quantiles.tolist()

    freq = "B" if req.interval == "daily" else "W-FRI"
    future_dates = pd.date_range(start=last_date, periods=max(horizons) + 1, freq=freq)[
        1:
    ]

    return ForecastResponse(
        ticker=req.ticker,
        interval=req.interval,
        predictions=[
            {
                "date": future_dates[horizon - 1].strftime("%Y-%m-%d"),
                "step": horizon,
                "quantiles": {
                    f"{quantile:.1f}": round(float(prediction_last[i, j]), 4)
                    for j, quantile in enumerate(quantiles)
                },
            }
            for i, horizon in enumerate(horizons)
        ],
        history=history,
    )
