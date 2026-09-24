import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ...errors import ApplicationError
from ...progress import progress
from .. import io, utils
from ..common.tasks import BaseTask
from ..datamodels import (
    CheckpointState,
    EpochLoss,
    Hyperparameters,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        best_checkpoint: Path,
        hyperparameters: Hyperparameters,
        latest_checkpoint: Path,
    ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.best_checkpoint = best_checkpoint
        self.latest_checkpoint = latest_checkpoint
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.history: list[EpochLoss] = []

        self.device = utils.get_model_device(self.model)
        self.task = task.to(self.device)
        self.use_amp = bool(
            hyperparameters.use_amp and self.device.type == "cuda"
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hyperparameters.learning_rate,
            eps=hyperparameters.adam_epsilon,
            weight_decay=hyperparameters.weight_decay,
        )

        floor_ratio = (
            hyperparameters.minimum_learning_rate / hyperparameters.learning_rate
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: max(
                floor_ratio,
                hyperparameters.learning_rate_factor
                ** (epoch // hyperparameters.learning_rate_period),
            ),
        )

        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        self.best_val_loss = float("inf")

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        description: str,
    ) -> list[EpochLoss]:
        for epoch in range(self.hyperparameters.max_epochs):
            if epoch < self.start_epoch:
                continue
            epoch_label = f"epoch {epoch + 1}/{self.hyperparameters.max_epochs}"
            training_loss = self.train_epoch(
                train_loader, f"Training {description} {epoch_label}"
            )
            val_loss = self.validate_epoch(
                validation_loader, f"Validating {description} {epoch_label}"
            )
            self.history.append(
                EpochLoss(
                    epoch=epoch + 1,
                    training_loss=training_loss,
                    validation_loss=val_loss,
                )
            )
            is_best = val_loss < (
                self.best_val_loss - self.hyperparameters.minimum_improvement
            )

            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            logger.info(
                "%s epoch %d/%d: training_loss=%.6f validation_loss=%.6f "
                "best_validation_loss=%.6f epochs_without_improvement=%d",
                description,
                epoch + 1,
                self.hyperparameters.max_epochs,
                training_loss,
                val_loss,
                self.best_val_loss,
                self.epochs_without_improvement,
            )

            self.save_checkpoint(epoch, self.latest_checkpoint)
            if is_best:
                self.save_checkpoint(epoch, self.best_checkpoint)
            if self.epochs_without_improvement >= self.hyperparameters.patience:
                logger.info(
                    "%s stopped early at epoch %d/%d after %d epochs without "
                    "improvement",
                    description,
                    epoch + 1,
                    self.hyperparameters.max_epochs,
                    self.epochs_without_improvement,
                )
                break
        return self.history

    def validate_epoch(self, dataloader: DataLoader, description: str) -> float:
        self.model.eval()
        running_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            batches = progress.track(
                dataloader,
                description=description,
            )
            for batch in batches:
                if self.use_amp:
                    with autocast(
                        device_type=self.device.type,
                        dtype=utils.AMP_DTYPE,
                    ):
                        loss = self.task.validation_step(self.model, batch)
                else:
                    loss = self.task.validation_step(self.model, batch)
                batch_size = batch.volume.shape[0]
                running_loss += loss.item() * batch_size
                sample_count += batch_size

        if sample_count == 0:
            raise ApplicationError(
                category="Input data",
                summary="Validation cannot continue with an empty DataLoader",
                cause=f"No batches were produced for '{description}'",
                fix=(
                    "Check split sizes, subject masks, patch extraction, and "
                    "batch settings"
                ),
            )

        return running_loss / sample_count

    def load_latest_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.latest_checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        utils.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_without_improvement = checkpoint["epochs_without_improvement"]
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint.get("history", [])

    def train_epoch(self, dataloader: DataLoader, description: str) -> float:
        self.model.train()
        running_loss = 0.0
        sample_count = 0

        batches = progress.track(
            dataloader,
            description=description,
        )
        for batch in batches:
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
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ApplicationError(
                category="Input data",
                summary="Training cannot continue with an empty DataLoader",
                cause=f"No batches were produced for '{description}'",
                fix="Check subject masks, patch extraction, and batch settings",
            )
        self.scheduler.step()
        return running_loss / sample_count

    def save_checkpoint(self, epoch: int, path: Path) -> None:
        unwrapped_model = utils.unwrap_model(self.model)

        state: CheckpointState = {
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
        }

        io.save_checkpoint(state, path)

