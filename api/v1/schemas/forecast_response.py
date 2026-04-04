from typing import TypedDict

from pydantic import BaseModel


class HorizonQuantile(TypedDict):
    step: int
    quantiles: dict[str, float]


class History(TypedDict):
    date: str
    close: float


class ForecastResponse(BaseModel):
    predictions: list[HorizonQuantile]
    history: list[History] | None
