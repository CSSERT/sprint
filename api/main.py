from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import backend.models.bootstrap  # noqa: F401
from backend.services import DataService, ForecastingModelService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import v1.routers as v1


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = ForecastingModelService.from_pretrained(
        Path.cwd()
        / ".."
        / "artifacts"
        / "sprint.transformers.patchtst"
        / "2026-04-27_epochs-100",
    )
    app.state.data = DataService(horizons=app.state.model.model.horizons.tolist())
    yield


app = FastAPI(
    title="Sprint API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(v1.router)
