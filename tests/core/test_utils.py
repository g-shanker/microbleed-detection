import numpy as np
import pytest
import torch

from microbleednet.core import utils
from microbleednet.core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorTeacher,
)


def test_stack_volume_and_frst_returns_channel_first_input() -> None:
    volume = np.ones((2, 3, 4))
    frst = np.full((2, 3, 4), 2.0)

    result = utils.stack_volume_and_frst(volume, frst)

    assert result.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(result[0], volume)
    np.testing.assert_array_equal(result[1], frst)


def test_microbleed_probability_detaches_logits_before_conversion() -> None:
    logits = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True).squeeze(0)

    probability = utils.microbleed_probability(logits)

    assert isinstance(probability, np.ndarray)
    assert probability == pytest.approx(
        torch.softmax(logits, dim=0)[1].detach().numpy()
    )


def test_initialize_teacher_transfers_detector_feature_and_segmentor_state() -> None:
    detector = CandidateDetector()
    teacher = CandidateDiscriminatorTeacher()
    classifier_weight = teacher.classifier.fc_3.weight.detach().clone()

    utils.initialize_teacher_from_detector(detector, teacher)

    torch.testing.assert_close(
        teacher.feature_extractor.in_conv.layer[0].weight,
        detector.feature_extractor.in_conv.layer[0].weight,
    )
    torch.testing.assert_close(
        teacher.segmentor.out_conv.layer.weight,
        detector.segmentor.out_conv.layer.weight,
    )
    torch.testing.assert_close(
        teacher.classifier.fc_3.weight,
        classifier_weight,
    )


def test_initialize_teacher_rejects_missing_detector_keys() -> None:
    detector = CandidateDetector()
    teacher = CandidateDiscriminatorTeacher()
    del teacher.segmentor

    with pytest.raises(RuntimeError, match="keys missing in teacher"):
        utils.initialize_teacher_from_detector(detector, teacher)