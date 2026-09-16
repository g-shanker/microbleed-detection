from pathlib import Path

import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from .. import io, utils
from ..common.tasks import BaseTask
from ..datamodels import CheckpointState, EpochLoss, TrainingHyperparameters
from .evaluators import Evaluator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        best_checkpoint: Path,
        hyperparameters: TrainingHyperparameters,
    ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.best_checkpoint = best_checkpoint
        self.epochs_without_improvement = 0

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
    ) -> list[EpochLoss]:
        history = []
        for epoch in range(self.hyperparameters.max_epochs):
            training_loss = self.train_epoch(train_loader)
            val_loss = self.evaluator.validation_loss(validation_loader)
            history.append(
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

            if is_best:
                self.save_checkpoint(epoch)
            if self.epochs_without_improvement >= self.hyperparameters.patience:
                break
        return history

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

        io.save_checkpoint(state, self.best_checkpoint)

