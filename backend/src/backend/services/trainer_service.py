import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import ForecastingModelService


@dataclass
class TrainerService:
    forecaster: ForecastingModelService
    optimizer: optim.Optimizer | None = None
    scheduler: Any | None = None
    epoch: int = 0
    global_step: int = 0

    def resume_from_checkpoint(
        self,
        checkpoint_dir: Path | str,
    ) -> None:
        optimizer_path = Path(checkpoint_dir) / "optimizer.pt"
        schedule_path = Path(checkpoint_dir) / "scheduler.pt"
        trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"

        if self.forecaster is None or self.optimizer is None:
            raise ValueError()

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(
                optimizer_path,
                map_location=self.forecaster.device,
            )
            self.optimizer.load_state_dict(optimizer_state_dict)

        if self.scheduler is not None and schedule_path.exists():
            scheduler_state_dict = torch.load(
                schedule_path,
                map_location=self.forecaster.device,
            )
            self.scheduler.load_state_dict(scheduler_state_dict)

        if trainer_state_path.exists():
            trainer_state = json.loads(trainer_state_path.read_text())
            self.epoch = trainer_state.get("epoch", 0)
            self.global_step = trainer_state.get("global_step", 0)

    def save_checkpoint(
        self,
        checkpoint_dir: Path | str,
    ) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        optimizer_path = Path(checkpoint_dir) / "optimizer.pt"
        schedule_path = Path(checkpoint_dir) / "scheduler.pt"
        trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"

        self.forecaster.save_pretrained(Path(checkpoint_dir))

        if self.optimizer is not None:
            optimizer_state_dict = self.optimizer.state_dict()
            torch.save(optimizer_state_dict, optimizer_path)

        if self.scheduler is not None:
            try:
                scheduler_state_dict = self.scheduler.state_dict()
                torch.save(scheduler_state_dict, schedule_path)
            except Exception:
                pass

        trainer_state = {"epoch": self.epoch, "global_step": self.global_step}
        trainer_state_path.write_text(json.dumps(trainer_state))

    def train(
        self,
        data_loader: DataLoader,
        *,
        epochs: int = 10,
        criterion: nn.Module | None = None,
        save_every: int | None = None,
        save_dir: Path | str | None = None,
    ) -> None:
        for _ in range(epochs):
            self.epoch += 1

            for batch in data_loader:
                self.forecaster.model.train()

                x, y = (t.to(self.forecaster.device) for t in batch)

                y_hat = self.forecaster.model(x)
                loss = y_hat if criterion is None else criterion(y_hat, y)

                if isinstance(loss, torch.Tensor):
                    loss.backward()

                    if self.optimizer is not None:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        try:
                            self.scheduler.step()
                        except Exception:
                            pass

                    self.global_step += 1

                if (
                    save_every is not None
                    and save_dir is not None
                    and self.global_step % save_every == 0
                ):
                    checkpoint_dir = Path(save_dir) / f"checkpoint-{self.global_step}"
                    self.save_checkpoint(checkpoint_dir)
