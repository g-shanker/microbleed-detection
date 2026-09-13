import torch
import torch.nn as nn

from microbleednet.core.common import tasks
from microbleednet.core.dataloading.datasets import (
    ClassificationBatch,
    SegmentationBatch,
    SegmentationClassificationBatch,
)


class SegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        shape = (volume.shape[0], 2, *volume.shape[2:])
        return self.bias + torch.zeros(shape, device=volume.device)


class TeacherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (volume.shape[0], 2, *volume.shape[2:])
        segmentation = self.bias + torch.zeros(shape, device=volume.device)
        classification = self.bias + torch.zeros(
            (volume.shape[0], 2), device=volume.device
        )
        return segmentation, classification


class StudentModel(SegmentationModel):
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.bias + torch.zeros(
            (volume.shape[0], 2), device=volume.device
        )


def test_tasks_compute_training_and_validation_losses(monkeypatch) -> None:
    volume = torch.zeros(2, 2, 2, 2, 2)
    mask = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    labels = torch.tensor([0, 1])

    segmentation_task = tasks.SegmentationTask()
    segmentation_batch = SegmentationBatch(volume, mask)
    assert torch.isfinite(
        segmentation_task.validation_step(
            SegmentationModel(), segmentation_batch
        )
    )

    combined_task = tasks.SegmentationClassificationTask()
    combined_batch = SegmentationClassificationBatch(volume, mask, labels)
    assert torch.isfinite(
        combined_task.validation_step(TeacherModel(), combined_batch)
    )

    teacher = TeacherModel()
    student_task = tasks.KnowledgeDistillationClassificationTask(teacher)
    classification_batch = ClassificationBatch(volume, labels)
    assert torch.isfinite(
        student_task.validation_step(StudentModel(), classification_batch)
    )
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_base_task_requires_step_implementations() -> None:
    base = tasks.BaseTask()
    model = nn.Linear(1, 1)

    for step in (base.training_step, base.validation_step):
        try:
            step(model, object())
        except NotImplementedError:
            pass
        else:
            raise AssertionError("base task step must be abstract")