import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ...progress import progress
from .. import io, utils
from ..common.tasks import BaseTask
from ..datamodels import (
    CheckpointState,
    EpochLoss,
    Hyperparameters,
)
from .evaluators import Evaluator

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

        self.evaluator = Evaluator(
            self.model,
            self.task,
            use_amp=self.use_amp,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        description: str,
    ) -> list[EpochLoss]:
        for epoch in progress.track(
            range(self.hyperparameters.max_epochs), description
        ):
            if epoch < self.start_epoch:
                continue
            training_loss = self.train_epoch(train_loader)
            val_loss = self.evaluator.validation_loss(validation_loader)
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

            self.save_checkpoint(epoch, self.latest_checkpoint)
            if is_best:
                self.save_checkpoint(epoch, self.best_checkpoint)
            logger.debug(
                "Epoch %d: train_loss=%.6f, validation_loss=%.6f, "
                "best=%s, learning_rate=%.6g, patience=%d/%d.",
                epoch + 1,
                training_loss,
                val_loss,
                is_best,
                self.optimizer.param_groups[0]["lr"],
                self.epochs_without_improvement,
                self.hyperparameters.patience,
            )
            if self.epochs_without_improvement >= self.hyperparameters.patience:
                logger.info(
                    "Early stopping after epoch %d; validation loss did not "
                    "improve for %d epochs.",
                    epoch + 1,
                    self.hyperparameters.patience,
                )
                break
        return self.history

    def load_latest_checkpoint(self) -> None:
        if self.latest_checkpoint is None:
            raise ValueError("latest checkpoint path is not configured")
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
        logger.info(
            "Resumed training from %s at epoch %d with %d recorded epochs.",
            self.latest_checkpoint,
            self.start_epoch,
            len(self.history),
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
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
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot train on an empty DataLoader")
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
        logger.debug("Saved checkpoint for epoch %d to %s.", epoch + 1, path)

