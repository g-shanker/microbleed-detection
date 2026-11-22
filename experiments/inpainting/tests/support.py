"""Tiny deterministic inputs shared by the inpainting experiment tests."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
)

SHAPE = (14, 14, 3)
LESION = (7, 7, 1)
VESSEL = (7, 9, 1)
# Marked as a vessel by the fake 3D detector only, so tests can tell the
# strategies apart.
THROUGH_PLANE = (7, 9, 2)
AFFINE = np.diag([1.0, 1.0, 3.0, 1.0])


def base_volume() -> np.ndarray:
    """The volume the fake preprocessing hands to vessel inpainting."""
    volume = np.random.default_rng(0).random(SHAPE) * 0.1
    volume[VESSEL] = 1.0
    return volume


def save(path: Path, array: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return str(path)


def write_raw_dataset(
    tmp_path: Path, subject_ids: list[str], *, with_masks: bool = True
) -> Path:
    """Write a raw manifest whose subjects point at tiny real NIfTI files."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    subjects = [
        RawSubject(
            subject_id=subject_id,
            source_id="source",
            volume_path=save(
                tmp_path / f"{subject_id}-volume.nii.gz",
                np.ones(SHAPE, dtype=np.float32),
            ),
            mask_path=(
                save(
                    tmp_path / f"{subject_id}-mask.nii.gz",
                    np.zeros(SHAPE, dtype=np.uint8),
                )
                if with_masks
                else None
            ),
        )
        for subject_id in subject_ids
    ]
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[
            RawSource(
                input_dir=str(tmp_path),
                volume_pattern="{subject_id}-volume.nii.gz",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=subjects,
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    return dataset_dir


def read_subjects(dataset_dir: Path) -> list[RawSubject]:
    """Read back the subjects of a dataset written by ``write_raw_dataset``."""
    path = DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    return RawDatasetManifest.read(path).subjects


def write_config(tmp_path: Path, dataset: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "dataset_dir": dataset.as_posix(),
        "bias_field_correction": "fast",
        "output_dir": (tmp_path / "results").as_posix(),
        "iterations": [0, 2],
        "strategies": ["axial", "frangi3d"],
        "peak_min_distance": 1,
        "exclusion_radius": 2,
        "lesion_margin": 0,
        "recall_at": [1],
        "max_lesion_loss": 1.0,
    }
    values.update(overrides)
    path = tmp_path / "config.toml"
    path.write_text(
        "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items())
    )
    return path
