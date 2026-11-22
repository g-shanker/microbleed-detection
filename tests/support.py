"""Explicitly imported builders shared by repository tests."""

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from microbleednet.core.datamodels import Hyperparameters
from microbleednet.orchestration.configs import IndexDataConfig, TrainConfig
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    SplitManifest,
    TrainManifest,
    content_fingerprint,
)

DEFAULT_SHAPE = (2, 2, 2)


def nifti(
    shape: tuple[int, ...] = DEFAULT_SHAPE,
    *,
    value: float = 1,
    data: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    dtype: Any = None,
) -> nib.Nifti1Image:
    """Build an in-memory NIfTI image from an array or a constant-filled shape."""
    array = np.full(shape, value, dtype=dtype or np.float32) if data is None else data
    return nib.Nifti1Image(array, np.eye(4) if affine is None else affine)


def save_nifti(path: Path, **kwargs: Any) -> str:
    """Save a NIfTI image built by ``nifti`` and return its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti(**kwargs), path)
    return str(path)


def hyperparameters(**overrides: Any) -> Hyperparameters:
    """Build valid training hyperparameters with concise overrides."""
    values: dict[str, Any] = {
        "batch_size": 8,
        "max_epochs": 100,
        "patience": 20,
        "learning_rate": 1e-3,
        "adam_epsilon": 1e-4,
        "learning_rate_factor": 0.1,
        "learning_rate_period": 2,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.0,
        "minimum_improvement": 0.0,
        "use_amp": False,
    }
    values.update(overrides)
    return Hyperparameters(**values)


def raw_source(source_id: str = "source", **overrides: Any) -> RawSource:
    """Build a raw source; ``input_dir`` and patterns default to placeholders."""
    values: dict[str, Any] = {
        "input_dir": ".",
        "volume_pattern": "{subject_id}.nii.gz",
        "source_id": source_id,
        "modality": "QSM",
    }
    values.update(overrides)
    return RawSource(**values)


def raw_subjects(
    count: int = 1,
    *,
    source_id: str = "source",
    directory: Path | None = None,
    with_mask: bool = True,
    **nifti_kwargs: Any,
) -> list[RawSubject]:
    """Build raw subjects, writing real NIfTI files when ``directory`` is given."""
    subjects = []
    for index in range(count):
        subject_id = f"subject-{index}"
        volume_path, mask_path = f"volume-{index}", f"mask-{index}"
        if directory is not None:
            volume_path = save_nifti(
                directory / f"{subject_id}_volume.nii.gz", **nifti_kwargs
            )
            mask_path = (
                save_nifti(directory / f"{subject_id}_mask.nii.gz", **nifti_kwargs)
                if with_mask
                else None
            )
        subjects.append(
            RawSubject(
                subject_id=subject_id,
                source_id=source_id,
                volume_path=volume_path,
                mask_path=mask_path if with_mask else None,
            )
        )
    return subjects


def write_raw_manifest(
    dataset_dir: Path,
    subjects: list[RawSubject] | None = None,
    *,
    status: ManifestStatus = ManifestStatus.COMPLETE,
    **overrides: Any,
) -> RawDatasetManifest:
    """Write a raw dataset manifest into ``dataset_dir`` and return it."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "status": status,
        "sources": [raw_source()],
        "subjects": raw_subjects() if subjects is None else subjects,
    }
    values.update(overrides)
    manifest = RawDatasetManifest(**values)
    manifest.write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    return manifest


def preprocessed_subjects(
    count: int,
    directory: Path,
    *,
    variants: int = 1,
    with_masks: bool = True,
    write_files: bool = False,
    **nifti_kwargs: Any,
) -> list[PreprocessedSubject]:
    """Build preprocessed subjects whose paths are touched or written to disk."""
    directory.mkdir(parents=True, exist_ok=True)

    def make_file(name: str) -> str:
        path = directory / name
        if write_files:
            return save_nifti(path, **nifti_kwargs)
        path.touch()
        return str(path)

    subjects = []
    for index in range(count):
        subject_variants = [
            PreprocessedVariant(
                volume_path=make_file(f"volume-{index}-{variant}.nii.gz"),
                mask_path=(
                    make_file(f"mask-{index}-{variant}.nii.gz") if with_masks else None
                ),
                frst_path=make_file(f"frst-{index}-{variant}.nii.gz"),
            )
            for variant in range(variants)
        ]
        subjects.append(
            PreprocessedSubject(
                subject_id=f"subject-{index}",
                original_volume_path=subject_variants[0].volume_path,
                brain_mask_path=subject_variants[0].volume_path,
                bounding_box=((0, 1), (0, 1), (0, 1)),
                variants=subject_variants,
            )
        )
    return subjects


def write_preprocessed_manifest(
    dataset_dir: Path,
    subjects: list[PreprocessedSubject],
    *,
    status: ManifestStatus = ManifestStatus.COMPLETE,
    **overrides: Any,
) -> PreprocessedDatasetManifest:
    """Write a preprocessed dataset manifest into ``dataset_dir`` and return it."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "status": status,
        "subjects": subjects,
        "bias_field_correction": "fast",
        "raw_manifest_fingerprint": "test-raw-manifest",
    }
    values.update(overrides)
    manifest = PreprocessedDatasetManifest(**values)
    manifest.write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    return manifest


def write_training_dataset(tmp_path: Path, count: int = 7, **kwargs: Any) -> Path:
    """Write a complete preprocessed dataset under ``tmp_path / 'dataset'``."""
    dataset_dir = tmp_path / "dataset"
    subjects = preprocessed_subjects(count, tmp_path / "inputs", **kwargs)
    write_preprocessed_manifest(dataset_dir, subjects)
    return dataset_dir


def write_split_manifest(
    tmp_path: Path,
    train_ids: list[str],
    *,
    validation_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    **overrides: Any,
) -> SplitManifest:
    """Write a complete split for ``tmp_path / 'dataset'`` into the experiment."""
    dataset_dir = tmp_path / "dataset"
    preprocessed = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    )
    values: dict[str, Any] = {
        "status": ManifestStatus.COMPLETE,
        "dataset_dir": str(dataset_dir.resolve()),
        "preprocessed_manifest_fingerprint": content_fingerprint(preprocessed),
        "train_size": 0.7,
        "validation_size": 0.1,
        "test_size": 0.2,
        "train_subject_ids": train_ids,
        "validation_subject_ids": validation_ids or [],
        "test_subject_ids": test_ids or [],
    }
    values.update(overrides)
    manifest = SplitManifest(**values)
    manifest.write(
        ExperimentLayout(experiment_dir=tmp_path / "experiment").split_manifest_path()
    )
    return manifest


def train_config(
    dataset_dir: Path, experiment_dir: Path, **overrides: Any
) -> TrainConfig:
    """Build a valid CPU training config with concise overrides."""
    recipe = hyperparameters()
    values: dict[str, Any] = {
        "device": "cpu",
        "detector_candidate_threshold": 0.5,
        "detector_augmentation_factor": 1,
        "discriminator_augmentation_factor": 1,
        "num_workers": 0,
        "pin_memory": False,
        "detector_hyperparameters": recipe,
        "teacher_hyperparameters": recipe,
        "student_hyperparameters": recipe,
    }
    values.update(overrides)
    return TrainConfig(
        dataset_dir=dataset_dir, experiment_dir=experiment_dir, **values
    )


def train_manifest_for(
    config: TrainConfig,
    split_manifest_fingerprint: str,
    *,
    status: ManifestStatus = ManifestStatus.RUNNING,
    **overrides: Any,
) -> TrainManifest:
    """Build the train manifest that matches ``config``."""
    values: dict[str, Any] = {
        "status": status,
        "dataset_dir": str(config.dataset_dir.resolve()),
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "device": config.device,
        "seed": config.seed,
        "detector_candidate_threshold": config.detector_candidate_threshold,
        "detector_augmentation_factor": config.detector_augmentation_factor,
        "discriminator_augmentation_factor": config.discriminator_augmentation_factor,
        "validation_augmentation_factor": 1,
        "detector_patch_size": 48,
        "discriminator_patch_size": 24,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    values.update(overrides)
    return TrainManifest(**values)


def make_index_config(tmp_path: Path, **overrides: object) -> IndexDataConfig:
    """Build a valid index-data config with concise per-test overrides."""
    (tmp_path / "masks").mkdir(exist_ok=True)
    values: dict[str, object] = {
        "dataset_dir": tmp_path / "dataset",
        "input_dir": tmp_path,
        "volume_pattern": "{subject_id}_volume.nii.gz",
        "mask_dir": tmp_path / "masks",
        "mask_pattern": "{subject_id}_mask.nii.gz",
        "source_id": "test-source",
        "modality": "T2*-GRE",
    }
    values.update(overrides)
    return IndexDataConfig.model_validate(values)


def write_index_config(path: Path, **overrides: object) -> Path:
    """Write a flat index-data TOML config and return its path."""
    values: dict[str, object] = {
        "dataset_dir": path.parent / "dataset",
        "input_dir": path.parent,
        "volume_pattern": "{subject_id}_volume.nii.gz",
        "mask_dir": path.parent / "masks",
        "mask_pattern": "{subject_id}_mask.nii.gz",
        "source_id": "test-source",
        "modality": "T2*-GRE",
    }
    values.update(overrides)

    lines = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (str, Path)):
            literal = json.dumps(str(value))
        else:
            raise TypeError(f"unsupported config value: {value!r}")
        lines.append(f"{key} = {literal}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_volume_source(tmp_path: Path, name: str, subject_ids: list[str]) -> Path:
    """Create a source directory and matching empty mask files."""
    source_dir = tmp_path / name
    mask_dir = tmp_path / f"{name}_masks"
    source_dir.mkdir()
    mask_dir.mkdir(exist_ok=True)
    for subject_id in subject_ids:
        (source_dir / f"{subject_id}_volume.nii.gz").write_bytes(b"")
        (mask_dir / f"{subject_id}_mask.nii.gz").write_bytes(b"")
    return source_dir
