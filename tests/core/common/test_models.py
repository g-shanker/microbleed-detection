import torch
import torch.nn as nn

from microbleednet.constants import DETECTOR_PATCH_SIZE, DISCRIMINATOR_PATCH_SIZE
from microbleednet.core.common.layers import DoubleConv
from microbleednet.core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
    weight_init,
)


@torch.no_grad()
def test_detector_preserves_detector_patch_shape() -> None:
    size = DETECTOR_PATCH_SIZE

    logits = CandidateDetector().eval()(torch.zeros(1, 2, size, size, size))

    assert logits.shape == (1, 2, size, size, size)


@torch.no_grad()
def test_teacher_returns_segmentation_and_classification_logits() -> None:
    size = DISCRIMINATOR_PATCH_SIZE

    segmentation, classification = CandidateDiscriminatorTeacher().eval()(
        torch.zeros(1, 2, size, size, size)
    )

    assert segmentation.shape == (1, 2, size, size, size)
    assert classification.shape == (1, 2)


@torch.no_grad()
def test_student_returns_classification_logits() -> None:
    size = DISCRIMINATOR_PATCH_SIZE

    logits = CandidateDiscriminatorStudent().eval()(torch.zeros(1, 2, size, size, size))

    assert logits.shape == (1, 2)


@torch.no_grad()
def test_double_conv_preserves_spatial_shape() -> None:
    output = DoubleConv(2, 4, 3, 3).eval()(torch.zeros(1, 2, 4, 4, 4))

    assert output.shape == (1, 4, 4, 4, 4)


def test_weight_init_skips_missing_bias() -> None:
    linear = nn.Linear(2, 2, bias=False)

    weight_init(linear)

    assert linear.bias is None
