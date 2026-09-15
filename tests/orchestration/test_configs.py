from pathlib import Path

import pytest
from pydantic import ValidationError

from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.configs import (
    IndexDataConfig,
    InferConfig,
    PreprocessConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
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
                "mask_dir": tmp_path / "masks",
                "mask_pattern": "{subject_id}.nii.gz",
            }
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"volume_pattern": "volume.nii.gz"}, "volume_pattern"),
        (
            {
                "mask_dir": "masks",
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
    with pytest.raises(ValidationError, match="mask_dir|mask_pattern"):
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


def test_index_config_rejects_missing_mask_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="mask_dir"):
        make_index_config(tmp_path, mask_dir=tmp_path / "missing")


def test_preprocess_config_rejects_missing_dataset_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="dataset_dir"):
        PreprocessConfig(dataset_dir=tmp_path / "missing", augmentation_factor=1)


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
                variants=[
                    PreprocessedVariant(
                        volume_path=str(volume_path),
                        mask_path=str(mask_path),
                        frst_path=str(volume_path),
                    )
                ],
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


def test_train_config_defaults_preserve_training_recipe(tmp_path: Path) -> None:
    config = TrainConfig(
        dataset_dir=_training_dataset(tmp_path),
        experiment_dir=tmp_path / "experiment",
    )

    assert config.train_size == 0.7
    assert config.validation_size == 0.1
    assert config.test_size == 0.2
    assert config.seed is None
    assert config.detector_candidate_threshold == 0.5
    assert config.detector_augmentation_factor == 10
    assert config.discriminator_augmentation_factor == 5
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.training_settings.batch_size == 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_size": 0.5},
        {"validation_size": 0.0},
        {"test_size": 0.3},
        {"detector_candidate_threshold": 1.1},
        {"detector_augmentation_factor": 0},
        {"discriminator_augmentation_factor": 0},
        {"num_workers": -1},
    ],
)
def test_train_config_rejects_invalid_training_values(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(
            {
                "dataset_dir": _training_dataset(tmp_path),
                "experiment_dir": tmp_path / "experiment",
                **overrides,
            }
        )


def test_train_config_accepts_custom_training_settings_and_pin_memory(
    tmp_path: Path,
) -> None:
    config = TrainConfig.model_validate(
        {
            "dataset_dir": _training_dataset(tmp_path),
            "experiment_dir": tmp_path / "experiment",
            "pin_memory": True,
            "training_settings": {"batch_size": 4, "max_epochs": 12},
        }
    )

    assert config.pin_memory is True
    assert config.training_settings.batch_size == 4
    assert config.training_settings.max_epochs == 12


def test_train_config_requires_split_sizes_to_sum_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        TrainConfig(
            dataset_dir=_training_dataset(tmp_path),
            experiment_dir=tmp_path / "experiment",
            train_size=0.6,
            validation_size=0.1,
            test_size=0.1,
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
        PreprocessConfig(dataset_dir=tmp_path, augmentation_factor=1)

    manifest_path = DatasetLayout(dataset_dir=tmp_path).raw_manifest_path()
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="not a versioned manifest"):
        PreprocessConfig(dataset_dir=tmp_path, augmentation_factor=1)


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
            variants=[
                PreprocessedVariant(
                    volume_path=str(tmp_path / f"missing-volume-{index}"),
                    mask_path=str(tmp_path / f"missing-mask-{index}"),
                    frst_path=str(tmp_path / f"missing-frst-{index}"),
                )
            ],
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


def test_infer_config_accepts_subjects_without_rechecking_variant_paths(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    for stage in ("detector", "student"):
        checkpoint = ExperimentLayout(
            experiment_dir=experiment_dir
        ).best_checkpoint_path(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    subject = PreprocessedSubject(
        subject_id="subject-1",
        variants=[
            PreprocessedVariant(
                volume_path=str(tmp_path / "missing-volume.nii.gz"),
                mask_path=str(tmp_path / "missing-mask.nii.gz"),
                frst_path=str(tmp_path / "missing-frst.nii.gz"),
            )
        ],
    )

    config = InferConfig(subjects=[subject], experiment_dir=experiment_dir)

    assert config.subjects == [subject]
    assert config.device == "cpu"


def test_infer_config_rejects_missing_student_checkpoint(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    detector_checkpoint = ExperimentLayout(
        experiment_dir=experiment_dir
    ).best_checkpoint_path("detector")
    detector_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    detector_checkpoint.touch()

    subject = PreprocessedSubject(
        subject_id="subject-1",
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path="mask", frst_path="frst"
            )
        ],
    )

    with pytest.raises(ValidationError, match="student checkpoint does not exist"):
        InferConfig(subjects=[subject], experiment_dir=experiment_dir)


def test_target_centered_config_validates_threshold(tmp_path: Path) -> None:
    values = {
        "experiment_layout": ExperimentLayout(experiment_dir=tmp_path),
        "stage": "student",
        "split": "train",
        "subjects": [
            PreprocessedSubject(
                subject_id="subject",
                variants=[
                    PreprocessedVariant(
                        volume_path="volume.nii.gz",
                        mask_path="mask.nii.gz",
                        frst_path="frst.nii.gz",
                    )
                ],
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
        "experiment_layout": ExperimentLayout(experiment_dir=tmp_path),
        "stage": "student",
        "split": "train",
        "subjects": [
            PreprocessedSubject(
                subject_id="subject",
                variants=[
                    PreprocessedVariant(
                        volume_path="volume.nii.gz",
                        mask_path="mask.nii.gz",
                        frst_path="frst.nii.gz",
                    )
                ],
            )
        ],
        "patch_size": 24,
        "augmentation_factor": 5,
        "probability_threshold": 0.5,
        "detector": object(),
    }

    with pytest.raises(TypeError, match="detector must be a CandidateDetector"):
        TargetCenteredPatchConfig.model_validate(values)
