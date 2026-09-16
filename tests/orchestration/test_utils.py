import torch

from microbleednet.orchestration import utils
from microbleednet.orchestration.manifests import (
    PreprocessedSubject,
    PreprocessedVariant,
)


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