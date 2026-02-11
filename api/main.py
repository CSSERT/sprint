from fastapi import FastAPI

from v1.routers import health

app = FastAPI(
    title="Sprint API",
    version="1.0.0",
)

app.include_router(health.router, prefix="/v1")
