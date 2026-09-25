import pickle
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from microbleednet.errors import ApplicationError
from microbleednet.orchestration import utils
from microbleednet.orchestration.manifests import (
    PreprocessedSubject,
    PreprocessedVariant,
)


def test_resolve_path_string_returns_absolute_string(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    assert utils.resolve_path_string(path) == str(path.resolve())


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("invalid checkpoint"),
        EOFError("empty checkpoint"),
        pickle.UnpicklingError("malformed checkpoint"),
    ],
)
def test_load_stage_checkpoint_wraps_expected_failure(
    tmp_path: Path, monkeypatch, failure: Exception
) -> None:
    checkpoint_path = tmp_path / "detector.pth"
    monkeypatch.setattr(
        utils.core_io,
        "load_model_weights",
        lambda *_: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(ApplicationError) as raised:
        utils.load_stage_checkpoint(
            nn.Linear(1, 1), checkpoint_path, "detector"
        )

    assert raised.value.category == "Checkpoint"
    assert raised.value.context == {
        "path": str(checkpoint_path),
        "stage": "detector",
    }
    assert str(failure) in str(raised.value)


def test_load_stage_checkpoint_preserves_application_error(
    tmp_path: Path, monkeypatch
) -> None:
    expected = ApplicationError(
        category="Checkpoint",
        summary="already structured",
    )
    monkeypatch.setattr(
        utils.core_io,
        "load_model_weights",
        lambda *_: (_ for _ in ()).throw(expected),
    )

    with pytest.raises(ApplicationError) as raised:
        utils.load_stage_checkpoint(
            nn.Linear(1, 1), tmp_path / "detector.pth", "detector"
        )

    assert raised.value is expected


def test_load_stage_checkpoint_preserves_unexpected_error(
    tmp_path: Path, monkeypatch
) -> None:
    expected = TypeError("unexpected checkpoint failure")
    monkeypatch.setattr(
        utils.core_io,
        "load_model_weights",
        lambda *_: (_ for _ in ()).throw(expected),
    )

    with pytest.raises(TypeError) as raised:
        utils.load_stage_checkpoint(
            nn.Linear(1, 1), tmp_path / "detector.pth", "detector"
        )

    assert raised.value is expected


def test_release_gpu_memory_collects_and_clears_cuda(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(utils.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    utils.release_gpu_memory()

    assert calls == ["gc", "cuda"]


def test_release_gpu_memory_skips_cuda_when_unavailable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(utils.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    utils.release_gpu_memory()

    assert calls == ["gc"]


def test_resolve_subjects_preserves_requested_id_order() -> None:
    subjects = [
        PreprocessedSubject(
            subject_id=subject_id,
            original_volume_path="original-volume",
            bounding_box=((0, 1), (0, 1), (0, 1)),
            variants=[
                PreprocessedVariant(
                    volume_path="volume",
                    mask_path="mask",
                    frst_path="frst",
                )
            ],
        )
        for subject_id in ("subject-1", "subject-2")
    ]

    resolved = utils.resolve_subjects(subjects, ["subject-2", "subject-1"])

    assert [subject.subject_id for subject in resolved] == [
        "subject-2",
        "subject-1",
    ]