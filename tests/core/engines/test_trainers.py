from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.dataloading.datasets import SegmentationBatch
from microbleednet.core.datamodels import EpochLoss, Hyperparameters
from microbleednet.core.engines.trainers import Trainer


class RegressionTask(BaseTask[SegmentationBatch]):
    def training_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        return (model(batch.volume) - batch.mask.float()).square().mean()

    def validation_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        return self.training_step(model, batch)


class BatchDataset(Dataset[SegmentationBatch]):
    def __init__(self, batches: list[SegmentationBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> SegmentationBatch:
        return self.batches[index]


def _hyperparameters(**overrides) -> Hyperparameters:
    values = {
        "batch_size": 8,
        "max_epochs": 100,
        "patience": 20,
        "learning_rate": 1e-3,
        "adam_epsilon": 1e-4,
        "learning_rate_factor": 0.1,
        "learning_rate_period": 2,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.0,
        "minimum_improvement": 0.0,
        "use_amp": False,
    }
    values.update(overrides)
    return Hyperparameters(**values)


def _trainer(tmp_path: Path, **overrides) -> Trainer:
    hyperparameters = _hyperparameters(**overrides)
    return Trainer(
        nn.Linear(1, 1),
        RegressionTask(),
        tmp_path / "best.pth",
        hyperparameters,
        tmp_path / "latest.pth",
    )


def test_fit_saves_and_restores_latest_checkpoint(tmp_path: Path, monkeypatch) -> None:
    latest_checkpoint = tmp_path / "latest.pth"
    trainer = Trainer(
        nn.Linear(1, 1),
        RegressionTask(),
        tmp_path / "best.pth",
        _hyperparameters(max_epochs=2, patience=5),
        latest_checkpoint,
    )
    monkeypatch.setattr(
        trainer, "train_epoch", lambda loader, description: 2.0
    )
    validation_losses = iter([1.0, 2.0])
    monkeypatch.setattr(
        trainer,
        "validate_epoch",
        lambda loader, description: next(validation_losses),
    )
    empty_loader = DataLoader(BatchDataset([]), batch_size=None)

    history = trainer.fit(empty_loader, empty_loader, "Training detector")

    assert latest_checkpoint.is_file()
    assert [entry["epoch"] for entry in history] == [1, 2]
    resumed = Trainer(
        nn.Linear(1, 1),
        RegressionTask(),
        tmp_path / "best-resumed.pth",
        _hyperparameters(max_epochs=3, patience=5),
        latest_checkpoint,
    )
    resumed.load_latest_checkpoint()

    assert resumed.start_epoch == 2
    assert resumed.history == history
    assert resumed.best_val_loss == 1.0
    assert resumed.epochs_without_improvement == 1


def test_train_epoch_updates_model_and_learning_rate(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_epochs=1)
    batches = [
        SegmentationBatch(torch.ones(2, 1), torch.zeros(2, 1)),
    ]
    loader = DataLoader(BatchDataset(batches), batch_size=None)
    linear = cast(nn.Linear, trainer.model)
    initial_weight = linear.weight.detach().clone()

    training_loss = trainer.train_epoch(loader, "Training detector epoch 1/1")

    assert not torch.equal(initial_weight, linear.weight)
    assert training_loss >= 0
    assert trainer.optimizer.param_groups[0]["eps"] == 1e-4
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3


def test_fit_stops_at_patience_and_saves_best(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path, max_epochs=10, patience=2)
    monkeypatch.setattr(
        trainer, "train_epoch", lambda loader, description: 2.0
    )
    losses = iter([1.0, 1.0, 1.0])
    monkeypatch.setattr(
        trainer,
        "validate_epoch",
        lambda loader, description: next(losses),
    )

    empty_loader = DataLoader(BatchDataset([]), batch_size=None)
    history = trainer.fit(empty_loader, empty_loader, "Training detector")

    assert trainer.best_val_loss == 1.0
    assert trainer.epochs_without_improvement == 2
    assert history == [
        EpochLoss(epoch=1, training_loss=2.0, validation_loss=1.0),
        EpochLoss(epoch=2, training_loss=2.0, validation_loss=1.0),
        EpochLoss(epoch=3, training_loss=2.0, validation_loss=1.0),
    ]
    assert torch.load(tmp_path / "best.pth", weights_only=True)["epoch"] == 0
    assert torch.load(tmp_path / "latest.pth", weights_only=True)["epoch"] == 2


def test_fit_skips_epochs_before_start_epoch(tmp_path: Path, monkeypatch) -> None:
    trainer = _trainer(tmp_path, max_epochs=2)
    trainer.start_epoch = 1
    monkeypatch.setattr(
        trainer, "train_epoch", lambda loader, description: 2.0
    )
    monkeypatch.setattr(
        trainer, "validate_epoch", lambda loader, description: 1.0
    )

    empty_loader = DataLoader(BatchDataset([]), batch_size=None)
    trainer.fit(empty_loader, empty_loader, "Training detector")

    assert [entry["epoch"] for entry in trainer.history] == [2]


def test_train_epoch_rejects_empty_loader(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_epochs=1)

    try:
        trainer.train_epoch(
            DataLoader(BatchDataset([]), batch_size=None),
            "Training detector epoch 1/1",
        )
    except ValueError as error:
        assert str(error) == "cannot train on an empty DataLoader"
    else:
        raise AssertionError("empty training loader must fail")


def test_validate_epoch_averages_batches_and_rejects_empty_loader() -> None:
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model,
        RegressionTask(),
        Path("best.pth"),
        _hyperparameters(),
        Path("latest.pth"),
    )
    loader = DataLoader(
        BatchDataset(
            [
                SegmentationBatch(torch.ones(2, 1), torch.zeros(2, 1)),
                SegmentationBatch(torch.ones(1, 1), torch.ones(1, 1)),
            ]
        ),
        batch_size=None,
    )

    assert trainer.validate_epoch(loader, "Validating detector epoch 1/100") >= 0

    try:
        trainer.validate_epoch(
            DataLoader(BatchDataset([]), batch_size=None),
            "Validating detector epoch 1/100",
        )
    except ValueError as error:
        assert "empty DataLoader" in str(error)
    else:
        raise AssertionError("empty validation loader must fail")


def test_validate_epoch_uses_amp_context(monkeypatch) -> None:
    calls = []

    class Context:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, *args) -> None:
            calls.append("exit")

    monkeypatch.setattr(
        "microbleednet.core.engines.trainers.autocast",
        lambda **kwargs: Context(),
    )
    trainer = Trainer(
        nn.Linear(1, 1),
        RegressionTask(),
        Path("best.pth"),
        _hyperparameters(use_amp=True),
        Path("latest.pth"),
    )
    trainer.use_amp = True
    loader = DataLoader(
        BatchDataset([SegmentationBatch(torch.ones(1, 1), torch.zeros(1, 1))]),
        batch_size=None,
    )

    trainer.validate_epoch(loader, "Validating detector epoch 1/100")

    assert calls == ["enter", "exit"]


def test_load_latest_checkpoint_requires_configured_path(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.latest_checkpoint = None  # pyright: ignore[reportAttributeAccessIssue]

    try:
        trainer.load_latest_checkpoint()
    except ValueError as error:
        assert str(error) == "latest checkpoint path is not configured"
    else:
        raise AssertionError("missing latest checkpoint path must fail")


def test_trainer_handles_zero_epochs_and_amp_training(
    tmp_path: Path, monkeypatch
) -> None:
    zero_epoch_trainer = _trainer(tmp_path / "zero", max_epochs=0)
    empty_loader = DataLoader(BatchDataset([]), batch_size=None)
    assert (
        zero_epoch_trainer.fit(empty_loader, empty_loader, "Training detector")
        == []
    )
    assert zero_epoch_trainer.best_val_loss == float("inf")

    calls = []

    class Context:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, *args) -> None:
            calls.append("exit")

    trainer = _trainer(tmp_path / "amp", max_epochs=1)
    trainer.use_amp = True
    monkeypatch.setattr(
        "microbleednet.core.engines.trainers.autocast",
        lambda **kwargs: Context(),
    )
    loader = DataLoader(
        BatchDataset([SegmentationBatch(torch.ones(1, 1), torch.zeros(1, 1))]),
        batch_size=None,
    )

    trainer.train_epoch(loader, "Training detector epoch 1/1")

    assert calls == ["enter", "exit"]