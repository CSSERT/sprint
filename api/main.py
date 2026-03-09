from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import backend.models.bootstrap  # noqa: F401
from backend.services import DataService, ForecastingModelService
from fastapi import FastAPI

import v1.routers as v1


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = ForecastingModelService.from_pretrained(
        Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest",
    )
    app.state.data = DataService(horizons=app.state.model.model.horizons.tolist())
    yield


app = FastAPI(
    title="Sprint API",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(v1.router)
