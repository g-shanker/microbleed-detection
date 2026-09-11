from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from .. import io, utils
from ..common.tasks import BaseTask
from ..datamodels import CheckpointState
from .evaluators import Evaluator


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int = 8
    max_epochs: int = 100
    patience: int = 20
    learning_rate: float = 1e-3
    adam_epsilon: float = 1e-4
    learning_rate_factor: float = 0.1
    learning_rate_period: int = 2
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    minimum_improvement: float = 0.0
    use_amp: bool = False


TRAINING_SETTINGS = TrainingSettings()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        best_checkpoint: Path,
        settings: TrainingSettings,
    ):
        self.model = model
        self.settings = settings
        self.best_checkpoint = best_checkpoint
        self.epochs_without_improvement = 0

        self.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        self.device = utils.get_model_device(self.model)
        self.task = task.to(self.device)
        self.use_amp = bool(settings.use_amp and self.device.type == "cuda")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=settings.learning_rate,
            eps=settings.adam_epsilon,
            weight_decay=settings.weight_decay,
        )

        floor_ratio = settings.minimum_learning_rate / settings.learning_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: max(
                floor_ratio,
                settings.learning_rate_factor
                ** (epoch // settings.learning_rate_period),
            ),
        )

        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        self.best_val_loss = float("inf")

        self.evaluator = Evaluator(
            self.model,
            self.task,
            use_amp=self.use_amp,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> None:
        for epoch in range(self.settings.max_epochs):
            self.train_epoch(train_loader)
            val_loss = self.evaluator.validation_loss(validation_loader)
            is_best = val_loss < (
                self.best_val_loss - self.settings.minimum_improvement
            )

            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if is_best:
                self.save_checkpoint(epoch)
            if self.epochs_without_improvement >= self.settings.patience:
                break

    def train_epoch(self, dataloader: DataLoader) -> None:
        self.model.train()
        sample_count = 0

        for batch in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=utils.AMP_DTYPE):
                    loss = self.task.training_step(self.model, batch)
            else:
                loss = self.task.training_step(self.model, batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = batch.volume.shape[0]
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot train on an empty DataLoader")
        self.scheduler.step()

    def save_checkpoint(self, epoch: int) -> None:
        unwrapped_model = utils.unwrap_model(self.model)

        state: CheckpointState = {
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        io.save_checkpoint_atomic(state, self.best_checkpoint)

