from typing import TypedDict

from pydantic import BaseModel


class HorizonPrediction(TypedDict):
    date: str
    step: int
    quantiles: dict[str, float]


class History(TypedDict):
    date: str
    close: float


class ForecastResponse(BaseModel):
    ticker: str
    interval: str
    predictions: list[HorizonPrediction]
    history: list[History] | None
