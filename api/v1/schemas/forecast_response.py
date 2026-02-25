from typing import TypedDict

from pydantic import BaseModel


class HorizonQuantile(TypedDict):
    step: int
    quantiles: dict[str, float]


class ForecastResponse(BaseModel):
    predictions: list[HorizonQuantile]
