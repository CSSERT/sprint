from pathlib import Path

from backend.services import DataService, ForecastingModelService
from fastapi import APIRouter, Depends

from ..schemas import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


def get_data_service(req: ForecastRequest) -> DataService:
    return DataService({"interval": req.interval})


def get_model_service() -> ForecastingModelService:
    return ForecastingModelService.from_pretrained(
        Path.cwd() / ".." / "artifacts" / "sprint.rnn.lstm" / "latest",
    )


@router.post("/", response_model=ForecastResponse)
def forecast(
    req: ForecastRequest,
    data_service: DataService = Depends(get_data_service),
    model_service: ForecastingModelService = Depends(get_model_service),
) -> ForecastResponse:
    train_loader = data_service.get(req.ticker)
    predictions = []
    for x, _ in train_loader:
        predictions.extend(model_service.predict(x.to("cuda")).cpu().numpy().tolist())
    return ForecastResponse(predictions=[predictions[-1]])
