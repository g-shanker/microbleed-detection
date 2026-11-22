"""Load the inputs of the dilation experiment from a raw dataset.

Vessel dilation happens in the middle of preprocessing, so the saved
preprocessed volumes (already inpainted) cannot be reused. Instead each subject
runs the real preprocessing with vessel inpainting switched off, and the
volume that would have been inpainted is cached together with the undilated
vessel mask of every detection strategy.
"""

import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
from nibabel.affines import voxel_sizes

from microbleednet.core import io
from microbleednet.core.datamodels import (
    BiasFieldCorrection,
    Modality,
    PreprocessInput,
)
from microbleednet.core.engines import processor
from microbleednet.orchestration.manifests import RawSubject

from . import detectors
from .log import Log, silent

BASE_ARRAYS = ("volume", "lesions", "brain", "affine")


@dataclass(frozen=True)
class PreparedSubject:
    """Everything needed to inpaint one subject with any strategy and dilation."""

    volume: np.ndarray
    lesions: np.ndarray
    brain: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    vessels: dict[str, np.ndarray]


def prepare_subject(
    subject: RawSubject,
    modality: Modality,
    bias_field_correction: BiasFieldCorrection,
    strategies: Sequence[str],
    cache_dir: Path,
    log: Log = silent,
) -> PreparedSubject:
    """Preprocess up to vessel inpainting and detect vessels with every strategy.

    The preprocessed arrays and each strategy's undilated vessel mask are cached,
    so a rerun only computes what is missing, such as a newly added strategy.
    Each step is reported to ``log``.
    """
    cache_path = cache_dir / bias_field_correction / f"{subject.subject_id}.npz"
    arrays = _load_cache(cache_path)
    if arrays is None:
        log("  preprocessing (not cached; runs FSL)")
        started = time.perf_counter()
        arrays = _preprocess(subject, modality, bias_field_correction)
        log(f"  preprocessing done in {time.perf_counter() - started:.1f}s")
    else:
        log("  preprocessing: cached")

    updated = False
    for strategy in strategies:
        key = f"vessels_{strategy}"
        if key in arrays:
            log(f"  detection {strategy}: cached")
            continue
        log(f"  detecting vessels: {strategy}")
        started = time.perf_counter()
        arrays[key] = detectors.DETECTORS[strategy](arrays["volume"])
        log(
            f"  detected {strategy} in {time.perf_counter() - started:.1f}s "
            f"({int(arrays[key].sum())} vessel voxels)"
        )
        updated = True
    if updated or not cache_path.exists():
        _write_cache(cache_path, arrays)
        log(f"  cache written: {cache_path}")
    return _prepared(arrays, strategies)


def load_prepared(path: Path, strategies: Sequence[str]) -> PreparedSubject | None:
    """Read a finished cache entry, or None if it is missing, partial or unreadable.

    Used to look at the subjects a run has finished so far, possibly while that
    run is still going, so it never preprocesses or detects anything.
    """
    try:
        arrays = _load_cache(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None
    if arrays is None or any(f"vessels_{s}" not in arrays for s in strategies):
        return None
    return _prepared(arrays, strategies)


def _prepared(
    arrays: dict[str, np.ndarray], strategies: Sequence[str]
) -> PreparedSubject:
    affine = arrays["affine"]
    x, y, z = (float(size) for size in voxel_sizes(affine))
    return PreparedSubject(
        volume=arrays["volume"],
        lesions=arrays["lesions"],
        brain=arrays["brain"],
        affine=affine,
        spacing=(x, y, z),
        vessels={s: arrays[f"vessels_{s}"] for s in strategies},
    )


def _write_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a cache file so that readers never see a partly written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **cast(dict[str, Any], arrays))
    temporary.replace(path)


def _load_cache(path: Path) -> dict[str, np.ndarray] | None:
    """Read a cache file, ignoring one written without the affine."""
    if not path.exists():
        return None
    with np.load(path) as cached:
        if any(name not in cached.files for name in BASE_ARRAYS):
            return None
        return {name: cached[name] for name in cached.files}


def _preprocess(
    subject: RawSubject,
    modality: Modality,
    bias_field_correction: BiasFieldCorrection,
) -> dict[str, np.ndarray]:
    """Run the pipeline's preprocessing, capturing the volume sent to inpainting."""
    captured: dict[str, np.ndarray] = {}

    def capture(volume: np.ndarray) -> np.ndarray:
        captured["volume"] = volume
        return volume

    raw_volume = io.load_volume(subject.volume_path)
    raw_mask = io.load_volume(subject.mask_path) if subject.mask_path else None
    with mock.patch.object(processor.inpaint_vessels, "apply", capture):
        output = processor.preprocess(
            PreprocessInput(raw_volume, raw_mask, modality, bias_field_correction)
        )
    assert output.mask is not None, "the experiment needs lesion masks"

    return {
        "volume": captured["volume"],
        "lesions": output.mask > 0,
        "brain": output.brain_mask > 0,
        "affine": np.asarray(output.affine, dtype=np.float64),
    }
