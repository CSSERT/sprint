from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from .types import (
    Checkpoint,
    ModelBuilder,
    ModelConfig,
    ModelRegistry,
)


@dataclass
class ModelService:
    model: nn.Module
    optimizer: optim.Optimizer | None = None
    criterion: nn.Module | None = None
    device: torch.device | str = "cuda"
    config: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,  # Everything after must be passed with keyword
        builder: ModelBuilder | None = None,
        registry: ModelRegistry | None = None,
        device: torch.device | str = "cuda",
    ) -> "ModelService":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config: ModelConfig = checkpoint.get("config", {})

        if builder is not None:
            model = builder(config)
        elif registry is not None:
            model_name = config.get("model_name")

            if model_name is None:
                raise ValueError("Work in Progress!")

            model_cls = registry.get(model_name)

            if model_cls is None:
                raise ValueError("Work in Progress!")

            model_kwargs = config.get("model_kwargs", {})
            model = model_cls(**model_kwargs)
        else:
            raise ValueError("Work in Progress!")

        state_dict = checkpoint.get("state_dict")
        service = cls(model=model, config=config)

        if state_dict is not None and service.optimizer is not None:
            service.optimizer.load_state_dict(state_dict)

        return service

    def save(
        self,
        checkpoint_dir: Path | str,
        *,
        filename: str = "latest.pt",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / filename

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state: Checkpoint = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
        }

        if self.optimizer is not None:
            checkpoint_state["state_dict"] = self.optimizer.state_dict()

        if extra_metadata is not None:
            checkpoint_state["metadata"] = extra_metadata

        torch.save(checkpoint_state, checkpoint_path)
        return checkpoint_path
