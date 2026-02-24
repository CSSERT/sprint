from typing import Any

from pydantic import BaseModel


class ForecastResponse(BaseModel):
    predictions: Any
