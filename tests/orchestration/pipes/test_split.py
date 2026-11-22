from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from microbleednet.orchestration.configs import SplitConfig
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import SplitManifest
from microbleednet.orchestration.pipes import split
from tests.support import (
    preprocessed_subjects,
    save_nifti,
    write_preprocessed_manifest,
)

SUBJECT_COUNT = 10


def _is_positive(subject_id: str) -> bool:
    return int(subject_id.removeprefix("subject-")) % 2 == 0


def _write_dataset(tmp_path: Path, maskless_index: int | None = None) -> Path:
    """Write a dataset whose even-numbered subjects contain a lesion mask."""
    subjects = preprocessed_subjects(SUBJECT_COUNT, tmp_path / "inputs")
    for index, subject in enumerate(subjects):
        variant = subject.variants[0]
        save_nifti(
            Path(str(variant.mask_path)),
            shape=(1, 1, 1),
            value=int(_is_positive(subject.subject_id)),
            dtype=np.uint8,
        )
        if index == maskless_index:
            subjects[index] = subject.model_copy(
                update={"variants": [variant.model_copy(update={"mask_path": None})]}
            )
    dataset_dir = tmp_path / "dataset"
    write_preprocessed_manifest(dataset_dir, subjects)
    return dataset_dir


def _config(tmp_path: Path, dataset_dir: Path, seed: int = 42) -> SplitConfig:
    return SplitConfig(
        dataset_dir=dataset_dir,
        experiment_dir=tmp_path / "experiment",
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        seed=seed,
    )


def _read_split(tmp_path: Path) -> SplitManifest:
    return SplitManifest.read(
        ExperimentLayout(experiment_dir=tmp_path / "experiment").split_manifest_path()
    )


def test_execute_partitions_all_subjects_with_stratified_lesion_balance(
    tmp_path: Path,
) -> None:
    split.execute(_config(tmp_path, _write_dataset(tmp_path)))

    manifest = _read_split(tmp_path)
    partitions = [
        manifest.train_subject_ids,
        manifest.validation_subject_ids,
        manifest.test_subject_ids,
    ]
    assert manifest.seed == 42
    assert [len(ids) for ids in partitions] == [6, 2, 2]
    assert sorted(sum(partitions, [])) == [
        f"subject-{index}" for index in range(SUBJECT_COUNT)
    ]
    for ids in partitions:
        assert sum(map(_is_positive, ids)) / len(ids) == 0.5


def test_execute_overwrites_previous_split_with_new_seed(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)
    split.execute(_config(tmp_path, dataset_dir, seed=42))
    first = _read_split(tmp_path)

    split.execute(_config(tmp_path, dataset_dir, seed=7))
    second = _read_split(tmp_path)

    assert (first.seed, second.seed) == (42, 7)
    assert first.train_subject_ids != second.train_subject_ids


def test_split_rejects_subjects_without_masks(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path, maskless_index=3)

    with pytest.raises(ValidationError, match="Required subject masks are missing"):
        _config(tmp_path, dataset_dir)

    assert (
        not ExperimentLayout(experiment_dir=tmp_path / "experiment")
        .split_manifest_path()
        .exists()
    )
