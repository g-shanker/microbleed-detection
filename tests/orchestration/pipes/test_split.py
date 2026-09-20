from pathlib import Path

import pytest
from pydantic import ValidationError

from microbleednet.orchestration.configs import SplitConfig
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    SplitManifest,
    timestamp,
)
from microbleednet.orchestration.pipes import split


def _write_preprocessed_manifest(
    dataset_dir: Path, count: int = 10, maskless_subject_id: str | None = None
) -> None:
    subjects = [
        PreprocessedSubject(
            subject_id=f"subject-{index}",
            variants=[
                PreprocessedVariant(
                    volume_path=f"volume-{index}",
                    mask_path=(
                        None
                        if f"subject-{index}" == maskless_subject_id
                        else f"mask-{index}"
                    ),
                    frst_path=f"frst-{index}",
                )
            ],
        )
        for index in range(count)
    ]
    now = timestamp()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=subjects,
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())


def test_split_execute_writes_seeded_manifest_and_overwrites_it(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_preprocessed_manifest(dataset_dir)
    experiment_dir = tmp_path / "experiment"

    config = SplitConfig(
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        seed=42,
    )
    split.execute(config)
    first = SplitManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).split_manifest_path()
    )

    split.execute(config.model_copy(update={"seed": 7}))
    second = SplitManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).split_manifest_path()
    )

    assert first.seed == 42
    assert second.seed == 7
    assert first.train_subject_ids != second.train_subject_ids
    assert sorted(
        second.train_subject_ids
        + second.validation_subject_ids
        + second.test_subject_ids
    ) == [f"subject-{index}" for index in range(10)]


def test_split_rejects_subjects_without_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_preprocessed_manifest(dataset_dir, maskless_subject_id="subject-3")
    experiment_dir = tmp_path / "experiment"

    with pytest.raises(
        ValidationError,
        match="subjects missing masks: subject-3",
    ):
        SplitConfig(
            dataset_dir=dataset_dir,
            experiment_dir=experiment_dir,
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            seed=42,
        )

    assert not ExperimentLayout(
        experiment_dir=experiment_dir
    ).split_manifest_path().exists()


