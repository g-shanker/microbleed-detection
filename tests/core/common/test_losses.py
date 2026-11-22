import pytest
import torch
import torch.nn.functional as F

from microbleednet.core.common import losses
from tests.support_cli_core import assert_close

TEACHER_LOGITS = torch.tensor([[2.0, 0.0]])
STUDENT_LOGITS = torch.tensor([[0.0, 2.0]])


@pytest.fixture
def volume_target() -> torch.Tensor:
    """Two 2x2x2 label volumes with a single microbleed voxel each."""
    return torch.tensor([[[[0, 1], [0, 0]], [[0, 0], [0, 0]]]] * 2)


def test_dice_loss_is_a_unit_interval_fraction(volume_target: torch.Tensor) -> None:
    probability = F.softmax(torch.zeros(2, 2, 2, 2, 2), dim=1)[:, 1]

    dice = losses.DiceLoss()(probability, volume_target == 1)

    assert 0 <= dice <= 1


def test_detector_loss_is_finite_and_differentiable(
    volume_target: torch.Tensor,
) -> None:
    logits = torch.zeros(2, 2, 2, 2, 2, requires_grad=True)

    detector = losses.DetectorLoss()(logits, volume_target)
    detector.backward()

    assert torch.isfinite(detector)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_matches_weighted_cross_entropy_without_focusing() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.2]])
    target = torch.tensor([0, 1, 1])
    weight = (1.0, 10.0)

    actual = losses.FocalLoss(weight, gamma=0.0)(logits, target)
    expected = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight))(logits, target)

    assert_close(actual, expected)


def test_focal_loss_down_weights_easy_voxels() -> None:
    logits = torch.tensor([[4.0, -4.0], [0.0, 0.0]])
    target = torch.tensor([0, 0])
    focal = losses.FocalLoss((1.0, 1.0), gamma=2.0)
    plain = losses.FocalLoss((1.0, 1.0), gamma=0.0)

    assert focal(logits, target) < plain(logits, target)


def test_detector_loss_combines_dice_and_focal_loss() -> None:
    logits = torch.tensor([[[[[0.3, -0.2]]], [[[0.1, 0.4]]]]])
    target = torch.tensor([[[[0, 1]]]])
    criterion = losses.DetectorLoss()

    expected = criterion.dice_loss(
        F.softmax(logits, dim=1)[:, 1], target == 1
    ) + criterion.focal_loss(logits, target)

    assert_close(criterion(logits, target), expected)


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

    assert_close(actual, expected)


def test_knowledge_distillation_loss_scales_by_temperature_square() -> None:
    criterion = losses.KnowledgeDistillationLoss()

    actual = criterion(TEACHER_LOGITS, STUDENT_LOGITS)
    expected = (
        F.kl_div(
            F.log_softmax(STUDENT_LOGITS / criterion.temperature, dim=1),
            F.softmax(TEACHER_LOGITS / criterion.temperature, dim=1),
            reduction="batchmean",
        )
        * criterion.temperature**2
    )

    assert_close(actual, expected)


def test_student_loss_uses_fixed_weights() -> None:
    target = torch.tensor([1])
    criterion = losses.DiscriminatorStudentLoss()

    actual = criterion(TEACHER_LOGITS, STUDENT_LOGITS, target)
    expected = 0.4 * criterion.cross_entropy_loss(STUDENT_LOGITS, target) + 0.6 * (
        criterion.knowledge_distillation_loss(TEACHER_LOGITS, STUDENT_LOGITS)
    )

    assert_close(actual, expected)
