import gc
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from ..core import io as core_io
from ..errors import ApplicationError
from .layouts import StageName
from .manifests import PreprocessedSubject


def resolve_path_string(path: Path) -> str:
    """Return an absolute path in its string representation."""
    return str(path.resolve())


def resolve_subjects(
    subjects: list[PreprocessedSubject], subject_ids: list[str]
) -> list[PreprocessedSubject]:
    """Select and order preprocessed subjects by identifier."""
    subjects_by_id = {subject.subject_id: subject for subject in subjects}
    return [subjects_by_id[subject_id] for subject_id in subject_ids]


def load_stage_checkpoint(
    model: nn.Module, checkpoint_path: Path, stage: StageName
) -> None:
    """Load stage weights while translating failures into application errors."""
    try:
        core_io.load_model_weights(model, checkpoint_path)
    except ApplicationError:
        raise
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise ApplicationError(
            category="Checkpoint",
            summary=f"Could not load the {stage} checkpoint",
            cause=str(error),
            fix=(
                f"Verify that the {stage} checkpoint exists and matches "
                "the expected model"
            ),
            context={"path": str(checkpoint_path), "stage": stage},
        ) from error


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


