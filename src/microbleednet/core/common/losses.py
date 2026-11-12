import torch
import torch.nn as nn
import torch.nn.functional as F

from ...constants import (
    DETECTOR_CLASS_WEIGHTS,
    DICE_SMOOTH,
    DISTILLATION_ALPHA,
    DISTILLATION_BETA,
    DISTILLATION_TEMPERATURE,
    FOREGROUND_CLASS,
    FOCAL_GAMMA,
)


class DiceLoss(nn.Module):
    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean soft Dice loss across a batch."""
        prediction = prediction.reshape(prediction.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)

        dice_coefficient = (2.0 * intersection + DICE_SMOOTH) / (union + DICE_SMOOTH)

        return 1.0 - dice_coefficient.mean()


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self):
        """Initialize distillation with the configured temperature."""
        super().__init__()
        self.temperature = DISTILLATION_TEMPERATURE

    def forward(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        """Measure KL divergence between softened teacher and student outputs."""
        teacher_predictions = F.softmax(teacher_logits / self.temperature, dim=1)
        student_predictions = F.log_softmax(student_logits / self.temperature, dim=1)

        divergence = F.kl_div(
            student_predictions, teacher_predictions, reduction="batchmean"
        )
        return divergence * self.temperature**2


class FocalLoss(nn.Module):
    def __init__(self, weight: tuple[float, ...], gamma: float):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", torch.tensor(weight))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(prediction.float(), dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        weight = self.weight[target]
        loss = -weight * (1 - log_pt.exp()) ** self.gamma * log_pt
        return loss.sum() / weight.sum()


class DetectorLoss(nn.Module):
    def __init__(self):
        """Build the combined Dice and weighted cross-entropy loss."""
        super().__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(DETECTOR_CLASS_WEIGHTS)
        )
        self.cross_entropy_loss = FocalLoss(
            weight=torch.tensor(DETECTOR_CLASS_WEIGHTS), 
            gamma=FOCAL_GAMMA,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combine foreground Dice loss with voxel-wise cross entropy."""
        cross_entropy_loss = self.cross_entropy_loss(logits, target)
        total_loss = cross_entropy_loss

        # has_foreground = (target == FOREGROUND_CLASS).flatten(start_dim=1).any(dim=1)
        # if has_foreground.any():
        #     # apply dice loss to only to samples that have foreground
        #     prediction = F.softmax(logits, dim=1)
        #     dice_loss = self.dice_loss(
        #         prediction[has_foreground, FOREGROUND_CLASS],
        #         target[has_foreground] == FOREGROUND_CLASS,
        #     )
        #     total_loss = total_loss + dice_loss

        # older method
        prediction = F.softmax(logits, dim=1)
        dice_loss = self.dice_loss(
            prediction[:, FOREGROUND_CLASS], target == FOREGROUND_CLASS
        )
        total_loss = total_loss + dice_loss

        return total_loss


class DiscriminatorTeacherLoss(nn.Module):
    """
    dice loss + weighted voxel-wise cross entropy loss + binary cross entropy
    """

    def __init__(self):
        """Build the teacher's segmentation and classification losses."""
        super().__init__()
        self.segmentation_loss = DetectorLoss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        classification_logits: torch.Tensor,
        classification_target: torch.Tensor,
        segmentation_logits: torch.Tensor,
        segmentation_target: torch.Tensor,
    ) -> torch.Tensor:
        """Sum teacher segmentation and classification losses."""
        segmentation_loss = self.segmentation_loss(
            segmentation_logits, segmentation_target
        )
        classification_loss = self.classification_loss(
            classification_logits,
            classification_target,
        )

        return segmentation_loss + classification_loss


class DiscriminatorStudentLoss(nn.Module):
    """
    alpha * cross entropy loss + beta * knowledge distillation loss
    """

    def __init__(self):
        """Build the weighted supervised and distillation losses."""
        super().__init__()
        self.alpha = DISTILLATION_ALPHA
        self.beta = DISTILLATION_BETA
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.knowledge_distillation_loss = KnowledgeDistillationLoss()

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Blend student cross entropy with teacher-guided distillation loss."""
        cross_entropy_loss = self.cross_entropy_loss(student_logits, target)
        knowledge_distillation_loss = self.knowledge_distillation_loss(
            teacher_logits,
            student_logits,
        )

        return self.alpha * cross_entropy_loss + self.beta * knowledge_distillation_loss
