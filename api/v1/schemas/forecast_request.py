from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    ticker: str
    interval: Literal["daily", "weekly"]
    lookback_days: int | None = Field(default=None, ge=1, le=365)
