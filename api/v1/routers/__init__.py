from fastapi import APIRouter

from . import health

router = APIRouter(prefix="/v1")

router.include_router(health.router)
