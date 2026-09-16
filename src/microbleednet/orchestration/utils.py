import gc
from pathlib import Path

import torch

from .manifests import PreprocessedSubject


def resolve_path_string(path: Path) -> str:
    return str(path.resolve())


def resolve_subjects(
    subjects: list[PreprocessedSubject], subject_ids: list[str]
) -> list[PreprocessedSubject]:
    subjects_by_id = {subject.subject_id: subject for subject in subjects}
    return [subjects_by_id[subject_id] for subject_id in subject_ids]


def release_gpu_memory() -> None:
    """Reclaim unreferenced GPU memory back to the CUDA driver.

    Runs a garbage collection so any just-dropped tensors are finalized, then
    returns the caching allocator's now-free blocks to the driver. A no-op-ish
    call off CUDA. Call this after dropping the last reference to a model or
    tensor that is no longer needed, so its memory is freed promptly rather than
    at the end of the enclosing scope.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


