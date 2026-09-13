import torch
import torch.nn.functional as F

from microbleednet.core.common import losses


def test_dice_and_detector_losses_are_finite() -> None:
    logits = torch.zeros(2, 2, 2, 2, 2, requires_grad=True)
    target = torch.tensor(
        [[[[0, 1], [0, 0]], [[0, 0], [0, 0]]]] * 2,
    )

    dice = losses.DiceLoss()(F.softmax(logits, dim=1)[:, 1], target == 1)
    detector = losses.DetectorLoss()(logits, target)

    assert 0 <= dice <= 1
    assert torch.isfinite(detector)
    detector.backward()


def test_teacher_loss_combines_segmentation_and_classification() -> None:
    segmentation_logits = torch.zeros(2, 2, 2, 2, 2)
    segmentation_target = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    classification_logits = torch.zeros(2, 2)
    classification_target = torch.tensor([0, 1])
    criterion = losses.DiscriminatorTeacherLoss()

    actual = criterion(
        classification_logits,
        classification_target,
        segmentation_logits,
        segmentation_target,
    )
    expected = criterion.segmentation_loss(
        segmentation_logits, segmentation_target
    ) + criterion.classification_loss(classification_logits, classification_target)

    torch.testing.assert_close(actual, expected)


def test_student_loss_uses_fixed_weights_without_temperature_square() -> None:
    teacher = torch.tensor([[2.0, 0.0]])
    student = torch.tensor([[0.0, 2.0]])
    target = torch.tensor([1])
    criterion = losses.DiscriminatorStudentLoss()

    actual = criterion(teacher, student, target)
    expected = 0.4 * criterion.cross_entropy_loss(student, target) + 0.6 * (
        criterion.knowledge_distillation_loss(teacher, student)
    )

    torch.testing.assert_close(actual, expected)