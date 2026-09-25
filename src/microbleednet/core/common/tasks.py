import torch
import torch.nn as nn

from ..dataloading.datasets import (
    ClassificationBatch,
    SegmentationBatch,
    SegmentationClassificationBatch,
)
from ..utils import get_model_device
from . import losses


class BaseTask[BatchType](nn.Module):
    def training_step(
        self, model: nn.Module, batch: BatchType
    ) -> torch.Tensor:
        """Compute a training loss for a batch."""
        raise NotImplementedError("BaseTask.training_step must be implemented")

    def validation_step(
        self, model: nn.Module, batch: BatchType
    ) -> torch.Tensor:
        """Compute a validation loss for a batch."""
        raise NotImplementedError("BaseTask.validation_step must be implemented")


class SegmentationTask(BaseTask[SegmentationBatch]):
    def __init__(self):
        """Initialize candidate segmentation training with detector loss."""
        super().__init__()
        self.criterion = losses.DetectorLoss()

    def training_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        """Compute segmentation loss for one training batch."""
        device = get_model_device(model)
        volume = batch.volume.to(device)
        mask = batch.mask.to(device)

        logits = model(volume)
        loss = self.criterion(logits, mask)

        return loss

    def validation_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        """Compute segmentation loss for one validation batch."""
        return self.training_step(model, batch)


class SegmentationClassificationTask(BaseTask[SegmentationClassificationBatch]):
    def __init__(self):
        """Initialize joint segmentation and classification training."""
        super().__init__()
        self.criterion = losses.DiscriminatorTeacherLoss()

    def training_step(
        self,
        model: nn.Module,
        batch: SegmentationClassificationBatch,
    ) -> torch.Tensor:
        """Compute joint segmentation and classification loss for a batch."""
        device = get_model_device(model)
        volume = batch.volume.to(device)
        mask = batch.mask.to(device)
        label = batch.label.to(device)

        segmentation_logits, classification_logits = model(volume)
        loss = self.criterion(
            classification_logits,
            label,
            segmentation_logits,
            mask,
        )

        return loss

    def validation_step(
        self,
        model: nn.Module,
        batch: SegmentationClassificationBatch,
    ) -> torch.Tensor:
        """Compute joint validation loss for a batch."""
        return self.training_step(model, batch)


class KnowledgeDistillationClassificationTask(BaseTask[ClassificationBatch]):
    def __init__(self, teacher_model: nn.Module):
        """Freeze a teacher model and initialize student distillation loss."""
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        self.criterion = losses.DiscriminatorStudentLoss()

    def training_step(
        self, model: nn.Module, batch: ClassificationBatch
    ) -> torch.Tensor:
        """Compute supervised and teacher-guided student loss for a batch."""
        device = get_model_device(model)
        volume = batch.volume.to(device)
        label = batch.label.to(device)

        with torch.no_grad():
            _, teacher_logits = self.teacher_model(volume)

        student_logits = model(volume)

        loss = self.criterion(teacher_logits, student_logits, label)

        return loss

    def validation_step(
        self, model: nn.Module, batch: ClassificationBatch
    ) -> torch.Tensor:
        """Compute student distillation loss for a validation batch."""
        return self.training_step(model, batch)
