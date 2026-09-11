from pathlib import Path

import pytest
from pydantic import ValidationError

from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.configs import (
    IndexDataConfig,
    PreprocessConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    timestamp,
)
from tests.support import make_index_config


@pytest.mark.parametrize("source_id", ["siteA", "site-a", "site_a", "123"])
def test_source_id_accepts_filesystem_safe_values(tmp_path, source_id: str) -> None:
    config = make_index_config(tmp_path, source_id=source_id)

    assert config.source_id == source_id


@pytest.mark.parametrize("source_id", ["site/A", "site A", "site@A", ""])
def test_source_id_rejects_unsafe_values(tmp_path, source_id: str) -> None:
    with pytest.raises(ValidationError, match="source_id"):
        make_index_config(tmp_path, source_id=source_id)


def test_source_id_is_required(tmp_path) -> None:
    with pytest.raises(ValidationError, match="source_id"):
        IndexDataConfig.model_validate(
            {
                "dataset_dir": tmp_path / "dataset",
                "input_dir": tmp_path,
                "volume_pattern": "{subject_id}.nii.gz",
                "label_dir": tmp_path / "masks",
                "mask_pattern": "{subject_id}.nii.gz",
            }
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"volume_pattern": "volume.nii.gz"}, "volume_pattern"),
        (
            {
                "label_dir": "labels",
                "mask_pattern": "mask.nii.gz",
            },
            "mask_pattern",
        ),
    ],
)
def test_patterns_require_subject_id_placeholder(
    tmp_path, overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        make_index_config(tmp_path, **overrides)


def test_pattern_accepts_multiple_subject_id_placeholders(tmp_path) -> None:
    pattern = "{subject_id}/{subject_id}_volume.nii.gz"

    config = make_index_config(tmp_path, volume_pattern=pattern)

    assert config.volume_pattern == pattern


def test_mask_inputs_are_required(tmp_path) -> None:
    with pytest.raises(ValidationError, match="label_dir|mask_pattern"):
        IndexDataConfig.model_validate(
            {
                "dataset_dir": tmp_path / "dataset",
                "input_dir": tmp_path,
                "volume_pattern": "{subject_id}.nii.gz",
                "source_id": "test-source",
            }
        )


def test_index_config_rejects_missing_input_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="input_dir"):
        make_index_config(tmp_path, input_dir=tmp_path / "missing")


def test_index_config_rejects_missing_label_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="label_dir"):
        make_index_config(tmp_path, label_dir=tmp_path / "missing")


def test_preprocess_config_rejects_missing_dataset_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="dataset_dir"):
        PreprocessConfig(dataset_dir=tmp_path / "missing")


def _training_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    subjects = []
    for index in range(7):
        volume_path = tmp_path / f"volume-{index}.nii.gz"
        mask_path = tmp_path / f"mask-{index}.nii.gz"
        volume_path.touch()
        mask_path.touch()
        subjects.append(
            PreprocessedSubject(
                subject_id=f"subject-{index}",
                volume_path=str(volume_path),
                mask_path=str(mask_path),
            )
        )
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=subjects,
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    return dataset_dir


def test_train_config_rejects_invalid_and_unavailable_devices(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    with pytest.raises(ValidationError, match="invalid device"):
        TrainConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            device="invalid",
        )

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValidationError, match="CUDA device is not available"):
        TrainConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            device="cuda",
        )


def test_train_config_requires_dataset_and_preprocessed_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="dataset_dir does not exist"):
        TrainConfig(
            dataset_dir=tmp_path / "missing",
            experiment_dir=tmp_path / "experiment",
        )

    dataset_dir = tmp_path / "empty-dataset"
    dataset_dir.mkdir()
    with pytest.raises(ValidationError, match="preprocessed manifest does not exist"):
        TrainConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
        )


def test_preprocess_config_requires_readable_raw_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="raw manifest does not exist"):
        PreprocessConfig(dataset_dir=tmp_path)

    manifest_path = DatasetLayout(dataset_dir=tmp_path).raw_manifest_path()
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="not a versioned manifest"):
        PreprocessConfig(dataset_dir=tmp_path)


def test_train_config_accepts_complete_manifest_without_training_policy(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=[],
    ).write(manifest_path)
    TrainConfig(dataset_dir=dataset_dir, experiment_dir=tmp_path / "experiment")

    missing_subjects = [
        PreprocessedSubject(
            subject_id=f"subject-{index}",
            volume_path=str(tmp_path / f"missing-volume-{index}"),
            mask_path=str(tmp_path / f"missing-mask-{index}"),
        )
        for index in range(7)
    ]
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=missing_subjects,
    ).write(manifest_path)
    TrainConfig(dataset_dir=dataset_dir, experiment_dir=tmp_path / "experiment")


def test_target_centered_config_validates_threshold(tmp_path: Path) -> None:
    values = {
        "patch_dir": tmp_path / "patches",
        "subjects": [
            PreprocessedSubject(
                subject_id="subject",
                volume_path="volume.nii.gz",
                mask_path="mask.nii.gz",
            )
        ],
        "patch_size": 24,
        "augmentation_factor": 5,
        "probability_threshold": 0.5,
        "detector": CandidateDetector(),
    }
    assert TargetCenteredPatchConfig.model_validate(values).probability_threshold == 0.5


def test_target_centered_config_rejects_invalid_detector(tmp_path: Path) -> None:
    values = {
        "patch_dir": tmp_path / "patches",
        "subjects": [
            PreprocessedSubject(
                subject_id="subject",
                volume_path="volume.nii.gz",
                mask_path="mask.nii.gz",
            )
        ],
        "patch_size": 24,
        "augmentation_factor": 5,
        "probability_threshold": 0.5,
        "detector": object(),
    }

    with pytest.raises(TypeError, match="detector must be a CandidateDetector"):
        TargetCenteredPatchConfig.model_validate(values)
