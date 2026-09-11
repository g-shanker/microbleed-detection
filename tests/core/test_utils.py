import numpy as np
import pytest
import torch

from microbleednet.core import utils
from microbleednet.core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorTeacher,
)


def test_microbleed_probability_detaches_logits_before_conversion() -> None:
    logits = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True).squeeze(0)

    probability = utils.microbleed_probability(logits)

    assert isinstance(probability, np.ndarray)
    assert probability == pytest.approx(
        torch.softmax(logits, dim=0)[1].detach().numpy()
    )


def test_append_frst_channel_preserves_volume_and_normalizes_response() -> None:
    volume = torch.arange(512, dtype=torch.float32).reshape(1, 1, 8, 8, 8)

    result = utils.append_frst_channel(volume)

    assert result.shape == (1, 2, 8, 8, 8)
    torch.testing.assert_close(result[:, :1], volume)
    assert torch.isfinite(result).all()
    assert 0 <= result[:, 1].min() <= result[:, 1].max() <= 1


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