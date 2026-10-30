from pathlib import Path

import nibabel as nib
import numpy as np
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
    subjects = []
    for index in range(count):
        subject_id = f"subject-{index}"
        mask_path = dataset_dir / f"mask-{index}.nii.gz"
        mask = np.zeros((1, 1, 1), dtype=np.uint8)
        if index % 2 == 0:
            mask[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        subjects.append(PreprocessedSubject(
            subject_id=f"subject-{index}",
            original_volume_path=f"volume-{index}",
            brain_mask_path=f"brain-mask-{index}",
            bounding_box=((0, 1), (0, 1), (0, 1)),
            variants=[
                PreprocessedVariant(
                    volume_path=f"volume-{index}",
                    mask_path=(
                        None
                        if subject_id == maskless_subject_id
                        else str(mask_path)
                    ),
                    frst_path=f"frst-{index}",
                )
            ],
        ))
    now = timestamp()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=subjects,
        bias_field_correction="fast",
        raw_manifest_fingerprint="test-raw-manifest",
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
    for subject_ids in (
        second.train_subject_ids,
        second.validation_subject_ids,
        second.test_subject_ids,
    ):
        positive_count = sum(
            int(subject_id.removeprefix("subject-")) % 2 == 0
            for subject_id in subject_ids
        )
        assert positive_count / len(subject_ids) == 0.5


def test_split_rejects_subjects_without_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_preprocessed_manifest(dataset_dir, maskless_subject_id="subject-3")
    experiment_dir = tmp_path / "experiment"

    with pytest.raises(
        ValidationError,
        match="Required subject masks are missing",
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


