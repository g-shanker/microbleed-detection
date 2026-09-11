import torch

from microbleednet.orchestration import utils


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