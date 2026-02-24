from fastapi import APIRouter

from . import forecast, health

router = APIRouter(prefix="/v1")

router.include_router(health.router)
router.include_router(forecast.router)
