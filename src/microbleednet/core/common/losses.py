import torch
import torch.nn as nn
import torch.nn.functional as F

DICE_SMOOTH = 1.0
FOREGROUND_CLASS = 1
DETECTOR_CLASS_WEIGHTS = (1.0, 10.0)
DISTILLATION_ALPHA = 0.4
DISTILLATION_BETA = 0.6
DISTILLATION_TEMPERATURE = 4.0


class DiceLoss(nn.Module):
    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction = prediction.reshape(prediction.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)

        dice_coefficient = (2.0 * intersection + DICE_SMOOTH) / (union + DICE_SMOOTH)

        return 1.0 - dice_coefficient.mean()


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = DISTILLATION_TEMPERATURE

    def forward(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        teacher_predictions = F.softmax(teacher_logits / self.temperature, dim=1)
        student_predictions = F.log_softmax(student_logits / self.temperature, dim=1)

        return F.kl_div(
            student_predictions, teacher_predictions, reduction="batchmean"
        )  # batchmean is for standard KL divergence


class DetectorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(DETECTOR_CLASS_WEIGHTS)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.softmax(logits, dim=1)
        dice_loss = self.dice_loss(
            prediction[:, FOREGROUND_CLASS], target == FOREGROUND_CLASS
        )
        cross_entropy_loss = self.cross_entropy_loss(logits, target)
        return dice_loss + cross_entropy_loss


class DiscriminatorTeacherLoss(nn.Module):
    """
    dice loss + weighted voxel-wise cross entropy loss + binary cross entropy
    """

    def __init__(self):
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
        cross_entropy_loss = self.cross_entropy_loss(student_logits, target)
        knowledge_distillation_loss = self.knowledge_distillation_loss(
            teacher_logits,
            student_logits,
        )

        return self.alpha * cross_entropy_loss + self.beta * knowledge_distillation_loss
