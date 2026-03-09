import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from safetensors.torch import (
    load_file as safetensors_load_file,
)
from safetensors.torch import (
    save_file as safetensors_save_file,
)
from torch.utils.data.dataloader import DataLoader

from ..models.registry import ModelRegistry, get_model_registry
from ..types.tickers import TickerId

ModelKwargs = dict[str, Any]


class ModelConfig(TypedDict, total=False):
    model_name: str
    model_kwargs: ModelKwargs


class UnknownModelError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model with name '{model_name}' does not exist in registry")


class ForecastingModelService:
    registry: ModelRegistry = get_model_registry()

    def __init__(
        self,
        model_name: str,
        model_kwargs: ModelKwargs | None = None,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        model_kwargs = model_kwargs or {}

        model_cls = self.registry.get(model_name)
        if model_cls is None:
            raise UnknownModelError(model_name)
        self.model = model_cls(**model_kwargs).to(device)

        self.device = device
        self.config: ModelConfig = {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: Path | str,
        *,
        device: torch.device | str = "cuda",
    ) -> "ForecastingModelService":
        safetensors_path = Path(pretrained_dir) / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError

        config_path = Path(pretrained_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError

        config: ModelConfig = json.loads(config_path.read_text())
        service = cls(
            config.get("model_name", ""),
            config.get("model_kwargs", {}),
            device=device,
        )

        model_state_dict = safetensors_load_file(safetensors_path)
        if model_state_dict is not None:
            service.model.load_state_dict(model_state_dict, strict=True)

        return service

    def save_pretrained(
        self,
        save_dir: Path | str,
    ) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config_path = Path(save_dir) / "config.json"
        config_path.write_text(json.dumps(self.config))

        safetensors_path = Path(save_dir) / "model.safetensors"
        model_state_dict = self.model.state_dict()
        safetensors_save_file(model_state_dict, safetensors_path)

    @torch.inference_mode()
    def predict(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, list[TickerId]]:
        self.model.eval()

        y_trues, y_hats, tickers = [], [], []

        for x, y, ticker in loader:
            y_hat = self.model(x.to(self.device), ticker.to(self.device))
            y_trues.append(y)
            y_hats.append(y_hat.cpu())
            tickers.append(ticker.cpu())

        y_trues = torch.cat(y_trues, dim=0).numpy()
        y_hats = torch.cat(y_hats, dim=0).numpy()
        tickers = torch.cat(tickers, dim=0).squeeze(-1).tolist()

        return y_hats, y_trues, tickers
