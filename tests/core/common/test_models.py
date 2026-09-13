import torch
import torch.nn as nn

from microbleednet.core.common.layers import DoubleConv
from microbleednet.core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
    weight_init,
)


def test_model_output_shapes_match_stage_patch_contracts() -> None:
    with torch.no_grad():
        detector = CandidateDetector().eval()
        detector_logits = detector(torch.zeros(1, 2, 48, 48, 48))

        teacher = CandidateDiscriminatorTeacher().eval()
        segmentation_logits, teacher_logits = teacher(
            torch.zeros(1, 2, 24, 24, 24)
        )

        student = CandidateDiscriminatorStudent().eval()
        student_logits = student(torch.zeros(1, 2, 24, 24, 24))

    assert detector_logits.shape == (1, 2, 48, 48, 48)
    assert segmentation_logits.shape == (1, 2, 24, 24, 24)
    assert teacher_logits.shape == student_logits.shape == (1, 2)


def test_double_conv_output_shape_and_biasless_initialization() -> None:
    layer = DoubleConv(2, 4, 3, 3).eval()
    output = layer(torch.zeros(1, 2, 4, 4, 4))
    linear = nn.Linear(2, 2, bias=False)

    weight_init(linear)

    assert output.shape == (1, 4, 4, 4, 4)
    assert linear.bias is None