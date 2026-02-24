import backend.models.bootstrap  # noqa: F401
import v1.routers as v1
from fastapi import FastAPI

app = FastAPI(
    title="Sprint API",
    version="1.0.0",
)

app.include_router(v1.router)
