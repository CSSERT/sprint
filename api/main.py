from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import backend.models.bootstrap  # noqa: F401
import v1.routers as v1
from backend.services import DataService, ForecastingModelService
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.data = DataService(
        interval="daily",
        horizons=[1, 5, 10, 20],
    )
    app.state.model = ForecastingModelService.from_pretrained(
        Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest",
    )
    yield


app = FastAPI(
    title="Sprint API",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(v1.router)
