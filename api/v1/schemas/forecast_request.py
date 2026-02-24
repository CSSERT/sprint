from typing import Literal

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    ticker: str
    interval: Literal["daily", "weekly"]
