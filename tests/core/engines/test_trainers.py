from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.dataloading.datasets import SegmentationBatch
from microbleednet.core.engines.evaluators import Evaluator
from microbleednet.core.engines.trainers import Trainer, TrainingSettings


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


def _trainer(tmp_path: Path, **overrides) -> Trainer:
    settings = TrainingSettings(**overrides)
    return Trainer(
        nn.Linear(1, 1),
        RegressionTask(),
        tmp_path / "best.pth",
        settings,
    )


def test_train_epoch_updates_model_and_learning_rate(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_epochs=1)
    batches = [
        SegmentationBatch(torch.ones(2, 1), torch.zeros(2, 1)),
    ]
    loader = DataLoader(BatchDataset(batches), batch_size=None)
    linear = cast(nn.Linear, trainer.model)
    initial_weight = linear.weight.detach().clone()

    trainer.train_epoch(loader)

    assert not torch.equal(initial_weight, linear.weight)
    assert trainer.optimizer.param_groups[0]["eps"] == 1e-4
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3


def test_fit_stops_at_patience_and_saves_best(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path, max_epochs=10, patience=2)
    monkeypatch.setattr(trainer, "train_epoch", lambda loader: None)
    losses = iter([1.0, 1.0, 1.0])
    monkeypatch.setattr(
        trainer.evaluator,
        "validation_loss",
        lambda loader: next(losses),
    )

    empty_loader = DataLoader(BatchDataset([]), batch_size=None)
    trainer.fit(empty_loader, empty_loader)

    assert trainer.best_val_loss == 1.0
    assert trainer.epochs_without_improvement == 2
    assert torch.load(tmp_path / "best.pth", weights_only=True)["epoch"] == 0
    assert not (tmp_path / "latest.pth").exists()


def test_train_epoch_rejects_empty_loader(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_epochs=1)

    try:
        trainer.train_epoch(DataLoader(BatchDataset([]), batch_size=None))
    except ValueError as error:
        assert str(error) == "cannot train on an empty DataLoader"
    else:
        raise AssertionError("empty training loader must fail")


def test_evaluator_averages_batches_and_rejects_empty_loader() -> None:
    model = nn.Linear(1, 1)
    evaluator = Evaluator(model, RegressionTask(), use_amp=False)
    loader = DataLoader(
        BatchDataset(
            [
                SegmentationBatch(torch.ones(2, 1), torch.zeros(2, 1)),
                SegmentationBatch(torch.ones(1, 1), torch.ones(1, 1)),
            ]
        ),
        batch_size=None,
    )

    assert evaluator.validation_loss(loader) >= 0

    try:
        evaluator.validation_loss(DataLoader(BatchDataset([]), batch_size=None))
    except ValueError as error:
        assert "empty DataLoader" in str(error)
    else:
        raise AssertionError("empty validation loader must fail")


def test_evaluator_uses_amp_context(monkeypatch) -> None:
    calls = []

    class Context:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, *args) -> None:
            calls.append("exit")

    monkeypatch.setattr(
        "microbleednet.core.engines.evaluators.autocast",
        lambda **kwargs: Context(),
    )
    evaluator = Evaluator(nn.Linear(1, 1), RegressionTask(), use_amp=True)
    loader = DataLoader(
        BatchDataset([SegmentationBatch(torch.ones(1, 1), torch.zeros(1, 1))]),
        batch_size=None,
    )

    evaluator.validation_loss(loader)

    assert calls == ["enter", "exit"]


def test_trainer_handles_zero_epochs_and_amp_training(
    tmp_path: Path, monkeypatch
) -> None:
    zero_epoch_trainer = _trainer(tmp_path / "zero", max_epochs=0)
    empty_loader = DataLoader(BatchDataset([]), batch_size=None)
    zero_epoch_trainer.fit(empty_loader, empty_loader)
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

    trainer.train_epoch(loader)

    assert calls == ["enter", "exit"]