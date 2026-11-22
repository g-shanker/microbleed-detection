from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.dataloading.datasets import SegmentationBatch
from microbleednet.core.datamodels import EpochLoss
from microbleednet.core.engines import trainers
from microbleednet.core.engines.trainers import Trainer
from tests.support import hyperparameters


class RegressionTask(BaseTask[SegmentationBatch]):
    def training_step(self, model: nn.Module, batch: SegmentationBatch) -> torch.Tensor:
        return (model(batch.volume) - batch.mask.float()).square().mean()

    def validation_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        return self.training_step(model, batch)


class BatchDataset(Dataset[SegmentationBatch]):
    def __init__(self, batches: tuple[SegmentationBatch, ...]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> SegmentationBatch:
        return self.batches[index]


def loader(*batches: SegmentationBatch) -> DataLoader:
    return DataLoader(BatchDataset(batches), batch_size=None)


def batch(rows: int = 1, target: float = 0.0) -> SegmentationBatch:
    """A batch whose identity-model loss is ``(1 - target) ** 2``."""
    return SegmentationBatch(torch.ones(rows, 1), torch.full((rows, 1), target))


def trainer(tmp_path: Path, **overrides) -> Trainer:
    """Trainer around ``y = x``, so losses are exact."""
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    return Trainer(
        model,
        RegressionTask(),
        tmp_path / "best.pth",
        hyperparameters(**overrides),
        tmp_path / "latest.pth",
    )


def stub_epochs(monkeypatch, subject: Trainer, validation_losses: list[float]) -> None:
    losses = iter(validation_losses)
    monkeypatch.setattr(subject, "train_epoch", lambda loader, description: 2.0)
    monkeypatch.setattr(
        subject, "validate_epoch", lambda loader, description: next(losses)
    )


@pytest.fixture
def autocast_events(monkeypatch) -> list:
    events = []

    @contextmanager
    def spy(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append("exit")

    monkeypatch.setattr(trainers, "autocast", spy)
    return events


def test_fit_saves_and_restores_latest_checkpoint(tmp_path: Path, monkeypatch) -> None:
    original = trainer(tmp_path, max_epochs=2, patience=5)
    stub_epochs(monkeypatch, original, [1.0, 2.0])

    history = original.fit(loader(), loader(), "detector")

    assert (tmp_path / "latest.pth").is_file()
    assert [entry["epoch"] for entry in history] == [1, 2]
    resumed = trainer(tmp_path, max_epochs=3, patience=5)
    resumed.load_latest_checkpoint()
    assert resumed.start_epoch == 2
    assert resumed.history == history
    assert resumed.best_val_loss == 1.0
    assert resumed.epochs_without_improvement == 1


def test_fit_stops_at_patience_and_saves_best(tmp_path: Path, monkeypatch) -> None:
    subject = trainer(tmp_path, max_epochs=10, patience=2)
    stub_epochs(monkeypatch, subject, [1.0, 1.0, 1.0])

    history = subject.fit(loader(), loader(), "detector")

    assert subject.best_val_loss == 1.0
    assert subject.epochs_without_improvement == 2
    assert history == [
        EpochLoss(epoch=epoch, training_loss=2.0, validation_loss=1.0)
        for epoch in (1, 2, 3)
    ]
    assert torch.load(tmp_path / "best.pth", weights_only=True)["epoch"] == 0
    assert torch.load(tmp_path / "latest.pth", weights_only=True)["epoch"] == 2


def test_fit_skips_epochs_before_start_epoch(tmp_path: Path, monkeypatch) -> None:
    subject = trainer(tmp_path, max_epochs=2)
    subject.start_epoch = 1
    stub_epochs(monkeypatch, subject, [1.0])

    subject.fit(loader(), loader(), "detector")

    assert [entry["epoch"] for entry in subject.history] == [2]


def test_fit_with_zero_epochs_returns_empty_history(tmp_path: Path) -> None:
    subject = trainer(tmp_path, max_epochs=0)

    assert subject.fit(loader(), loader(), "detector") == []
    assert subject.best_val_loss == float("inf")


def test_train_epoch_returns_loss_and_updates_model(tmp_path: Path) -> None:
    subject = trainer(tmp_path)
    linear = cast(nn.Linear, subject.model)
    initial_weight = linear.weight.detach().clone()

    training_loss = subject.train_epoch(loader(batch()), "Training detector")

    assert training_loss == pytest.approx(1.0)
    assert not torch.equal(initial_weight, linear.weight)
    assert subject.optimizer.param_groups[0]["eps"] == 1e-4


def test_train_epoch_steps_learning_rate_schedule(tmp_path: Path) -> None:
    subject = trainer(tmp_path, learning_rate_period=1)

    subject.train_epoch(loader(batch()), "Training detector")

    assert subject.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_validate_epoch_weights_batches_by_sample_count(tmp_path: Path) -> None:
    subject = trainer(tmp_path)

    validation_loss = subject.validate_epoch(
        loader(batch(rows=2), batch(rows=1, target=1.0)), "Validating detector"
    )

    assert validation_loss == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("train_epoch", "Training cannot continue with an empty DataLoader"),
        ("validate_epoch", "Validation cannot continue with an empty DataLoader"),
    ],
)
def test_epoch_rejects_empty_loader(tmp_path: Path, method: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(trainer(tmp_path), method)(loader(), "detector")


def test_train_epoch_rejects_non_finite_loss(tmp_path: Path) -> None:
    nan_batch = SegmentationBatch(torch.full((2, 1), torch.nan), torch.zeros(2, 1))

    with pytest.raises(ValueError, match="non-finite loss"):
        trainer(tmp_path).train_epoch(loader(nan_batch), "Training detector")


@pytest.mark.parametrize("method", ["train_epoch", "validate_epoch"])
def test_epoch_uses_amp_context_when_enabled(
    tmp_path: Path, autocast_events: list, method: str
) -> None:
    subject = trainer(tmp_path)
    subject.use_amp = True

    getattr(subject, method)(loader(batch()), "detector")

    assert autocast_events == [
        ("enter", {"device_type": "cpu", "dtype": torch.bfloat16}),
        "exit",
    ]
