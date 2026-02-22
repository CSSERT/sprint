from fastapi import FastAPI

import v1.routers as v1

app = FastAPI(
    title="Sprint API",
    version="1.0.0",
)

app.include_router(v1.router)
