import logging
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from microbleednet.errors import ApplicationError
from microbleednet.orchestration import utils
from tests.support import preprocessed_subjects


def _fail_checkpoint_load(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(utils.core_io, "load_model_weights", Mock(side_effect=error))


def test_create_rng_is_stable_for_matching_seed_and_spawn_key() -> None:
    draws = [utils.create_rng(42, 3, key).random(4) for key in (2, 2, 3)]

    np.testing.assert_array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])


def test_create_rng_accepts_no_seed() -> None:
    assert isinstance(utils.create_rng(), np.random.Generator)


def test_subject_failure_context_enriches_application_error(caplog) -> None:
    expected = ApplicationError(
        category="Input data",
        summary="Bad input",
        context={"path": "volume.nii.gz"},
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ApplicationError) as raised:
            with utils.subject_failure_context("subject-17", "preprocessing"):
                raise expected

    assert raised.value is expected
    assert expected.context == {
        "path": "volume.nii.gz",
        "subject_id": "subject-17",
        "stage": "preprocessing",
    }
    assert "subject_id=subject-17" in caplog.text
    assert "stage=preprocessing" in caplog.text


def test_subject_failure_context_logs_unexpected_error(caplog) -> None:
    expected = RuntimeError("unexpected failure")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as raised:
            with utils.subject_failure_context("subject-23", "inference"):
                raise expected

    assert raised.value is expected
    assert "subject_id=subject-23" in caplog.text
    assert "stage=inference" in caplog.text
    assert "unexpected failure" in caplog.text


def test_subject_failure_context_leaves_success_unlogged(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with utils.subject_failure_context("subject-42", "evaluation"):
            pass

    assert not caplog.records


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    checkpoint_path = tmp_path / "detector.pth"
    _fail_checkpoint_load(monkeypatch, failure)

    with pytest.raises(ApplicationError) as raised:
        utils.load_stage_checkpoint(nn.Linear(1, 1), checkpoint_path, "detector")

    assert raised.value.category == "Checkpoint"
    assert raised.value.context == {
        "path": str(checkpoint_path),
        "stage": "detector",
    }
    assert str(failure) in str(raised.value)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        ApplicationError(category="Checkpoint", summary="already structured"),
        TypeError("unexpected checkpoint failure"),
    ],
    ids=["application-error", "unexpected-error"],
)
def test_load_stage_checkpoint_preserves_unwrapped_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    _fail_checkpoint_load(monkeypatch, failure)

    with pytest.raises(type(failure)) as raised:
        utils.load_stage_checkpoint(nn.Linear(1, 1), tmp_path / "x.pth", "detector")

    assert raised.value is failure


@pytest.mark.parametrize(
    ("cuda_available", "expected_calls"),
    [(True, ["gc", "cuda"]), (False, ["gc"])],
)
def test_release_gpu_memory_collects_and_clears_cuda_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_calls: list[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(utils.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    utils.release_gpu_memory()

    assert calls == expected_calls


def test_resolve_subjects_preserves_requested_id_order(tmp_path: Path) -> None:
    subjects = preprocessed_subjects(2, tmp_path)

    resolved = utils.resolve_subjects(subjects, ["subject-1", "subject-0"])

    assert resolved == [subjects[1], subjects[0]]
