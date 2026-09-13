import torch
import torch.nn as nn

from ..dataloading.datasets import (
    ClassificationBatch,
    SegmentationBatch,
    SegmentationClassificationBatch,
)
from ..utils import append_frst_channel, get_model_device
from . import losses


class BaseTask[BatchType](nn.Module):
    def training_step(
        self, model: nn.Module, batch: BatchType
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the training_step method."
        )

    def validation_step(
        self, model: nn.Module, batch: BatchType
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the validation_step method."
        )


class SegmentationTask(BaseTask[SegmentationBatch]):
    def __init__(self):
        super().__init__()
        self.criterion = losses.DetectorLoss()

    def training_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        device = get_model_device(model)
        volume = batch.volume.to(device)
        mask = batch.mask.to(device)

        logits = model(append_frst_channel(volume))
        loss = self.criterion(logits, mask)

        return loss

    def validation_step(
        self, model: nn.Module, batch: SegmentationBatch
    ) -> torch.Tensor:
        return self.training_step(model, batch)


class SegmentationClassificationTask(BaseTask[SegmentationClassificationBatch]):
    def __init__(self):
        super().__init__()
        self.criterion = losses.DiscriminatorTeacherLoss()

    def training_step(
        self,
        model: nn.Module,
        batch: SegmentationClassificationBatch,
    ) -> torch.Tensor:
        device = get_model_device(model)
        volume = batch.volume.to(device)
        mask = batch.mask.to(device)
        label = batch.label.to(device)

        segmentation_logits, classification_logits = model(
            append_frst_channel(volume)
        )
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
        return self.training_step(model, batch)


class KnowledgeDistillationClassificationTask(BaseTask[ClassificationBatch]):
    def __init__(self, teacher_model: nn.Module):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        self.criterion = losses.DiscriminatorStudentLoss()

    def training_step(
        self, model: nn.Module, batch: ClassificationBatch
    ) -> torch.Tensor:
        device = get_model_device(model)
        volume = batch.volume.to(device)
        label = batch.label.to(device)

        volume = append_frst_channel(volume)

        with torch.no_grad():
            _, teacher_logits = self.teacher_model(volume)

        student_logits = model(volume)

        loss = self.criterion(teacher_logits, student_logits, label)

        return loss

    def validation_step(
        self, model: nn.Module, batch: ClassificationBatch
    ) -> torch.Tensor:
        return self.training_step(model, batch)
