from pydantic import BaseModel


class ForecastResponse(BaseModel):
    predictions: list[float]
