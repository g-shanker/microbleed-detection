import pytest
import torch
import torch.nn as nn

from microbleednet.core.common import tasks
from microbleednet.core.dataloading.datasets import (
    ClassificationBatch,
    SegmentationBatch,
    SegmentationClassificationBatch,
)
from tests.support_cli_core import assert_close

VOLUME = torch.zeros(2, 2, 2, 2, 2)
MASK = torch.zeros(2, 2, 2, 2, dtype=torch.long)
LABEL = torch.tensor([0, 1])


class StubModel(nn.Module):
    """Zero logits shifted by a trainable bias, one output per requested head."""

    def __init__(self, *heads: str) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.heads = heads

    def forward(self, volume: torch.Tensor):
        outputs = tuple(
            self.bias
            + torch.zeros(
                volume.shape[0],
                2,
                *(volume.shape[2:] if head == "segmentation" else ()),
            )
            for head in self.heads
        )
        return outputs[0] if len(outputs) == 1 else outputs


@pytest.fixture(params=["segmentation", "teacher", "student"])
def task_case(request: pytest.FixtureRequest):
    """A task, a model it can train, and a matching batch."""
    if request.param == "segmentation":
        return (
            tasks.SegmentationTask(),
            StubModel("segmentation"),
            SegmentationBatch(VOLUME, MASK),
        )
    if request.param == "teacher":
        return (
            tasks.SegmentationClassificationTask(),
            StubModel("segmentation", "classification"),
            SegmentationClassificationBatch(VOLUME, MASK, LABEL),
        )
    return (
        tasks.KnowledgeDistillationClassificationTask(
            StubModel("segmentation", "classification")
        ),
        StubModel("classification"),
        ClassificationBatch(VOLUME, LABEL),
    )


def test_training_step_returns_finite_differentiable_loss(task_case) -> None:
    task, model, batch = task_case

    loss = task.training_step(model, batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.bias.grad is not None


def test_validation_step_matches_training_step(task_case) -> None:
    task, model, batch = task_case

    assert_close(task.validation_step(model, batch), task.training_step(model, batch))


def test_distillation_task_freezes_teacher() -> None:
    teacher = StubModel("segmentation", "classification")

    tasks.KnowledgeDistillationClassificationTask(teacher)

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


@pytest.mark.parametrize("step", ["training_step", "validation_step"])
def test_base_task_requires_step_implementations(step: str) -> None:
    with pytest.raises(NotImplementedError):
        getattr(tasks.BaseTask(), step)(nn.Linear(1, 1), object())
