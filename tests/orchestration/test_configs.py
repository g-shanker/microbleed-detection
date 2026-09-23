from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from microbleednet.core.common.models import CandidateDetector
from microbleednet.core.datamodels import Hyperparameters
from microbleednet.orchestration.configs import (
    EvaluateConfig,
    IndexDataConfig,
    InferConfig,
    PreprocessConfig,
    SplitConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
    ensure_training_resume_state,
)
from microbleednet.orchestration.layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    DatasetLayout,
    ExperimentLayout,
)
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
    TrainStageManifest,
    content_fingerprint,
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


def test_modality_is_required(tmp_path) -> None:
    with pytest.raises(ValidationError, match="modality"):
        IndexDataConfig.model_validate(
            {
                "dataset_dir": tmp_path / "dataset",
                "input_dir": tmp_path,
                "volume_pattern": "{subject_id}.nii.gz",
                "mask_dir": tmp_path / "masks",
                "mask_pattern": "{subject_id}.nii.gz",
                "source_id": "test-source",
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


def test_mask_inputs_are_optional(tmp_path) -> None:
    config = IndexDataConfig.model_validate(
        {
            "dataset_dir": tmp_path / "dataset",
            "input_dir": tmp_path,
            "volume_pattern": "{subject_id}.nii.gz",
            "source_id": "test-source",
            "modality": "T2*-GRE",
        }
    )

    assert config.mask_dir is None
    assert config.mask_pattern is None


def test_mask_inputs_must_be_provided_together(tmp_path) -> None:
    with pytest.raises(ValidationError, match="provided together"):
        IndexDataConfig.model_validate(
            {
                "dataset_dir": tmp_path / "dataset",
                "input_dir": tmp_path,
                "volume_pattern": "{subject_id}.nii.gz",
                "mask_dir": tmp_path,
                "source_id": "test-source",
                "modality": "T2*-GRE",
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


def test_preprocess_config_ignores_existing_checkpoint_without_resume(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    raw_manifest = _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=str(tmp_path / "mask.nii.gz"),
    )
    PreprocessedDatasetManifest(
        status=ManifestStatus.RUNNING,
        created_at=timestamp(),
        updated_at=timestamp(),
        subjects=[],
        augmentation_factor=99,
        raw_manifest_fingerprint=content_fingerprint(raw_manifest),
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())

    config = PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)

    assert config.resume is False


def test_preprocess_config_resume_requires_existing_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=str(tmp_path / "mask.nii.gz"),
    )

    with pytest.raises(ValidationError, match="preprocessed manifest does not exist"):
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1, resume=True)


def test_preprocess_config_resume_accepts_running_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    raw_manifest = _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=str(tmp_path / "mask.nii.gz"),
    )
    PreprocessedDatasetManifest(
        status=ManifestStatus.RUNNING,
        created_at=timestamp(),
        updated_at=timestamp(),
        subjects=[],
        augmentation_factor=1,
        raw_manifest_fingerprint=content_fingerprint(raw_manifest),
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())

    config = PreprocessConfig(
        dataset_dir=dataset_dir, augmentation_factor=1, resume=True
    )

    assert config.resume is True


def test_preprocess_config_resume_rejects_complete_manifest(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    raw_manifest = _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=str(tmp_path / "mask.nii.gz"),
    )
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=timestamp(),
        updated_at=timestamp(),
        subjects=[],
        augmentation_factor=3,
        raw_manifest_fingerprint=content_fingerprint(raw_manifest),
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())

    with pytest.raises(
        ValidationError,
        match="the preprocessing run is already complete",
    ):
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=3, resume=True)


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
                original_volume_path=str(volume_path),
                bounding_box=((0, 1), (0, 1), (0, 1)),
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
        raw_manifest_fingerprint="test-raw-manifest",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    return dataset_dir


def _complete_split(tmp_path: Path, subject_ids: list[str]) -> None:
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=tmp_path / "dataset").preprocessed_manifest_path()
    )
    SplitManifest(
        status=ManifestStatus.COMPLETE,
        created_at=timestamp(),
        updated_at=timestamp(),
        dataset_dir=str((tmp_path / "dataset").resolve()),
        preprocessed_manifest_fingerprint=content_fingerprint(preprocessed_manifest),
        train_size=0.7,
        validation_size=0.1,
        test_size=0.2,
        train_subject_ids=subject_ids,
        validation_subject_ids=[],
        test_subject_ids=[],
    ).write(
        ExperimentLayout(experiment_dir=tmp_path / "experiment").split_manifest_path()
    )


def _train_config(
    dataset_dir: Path, experiment_dir: Path, **overrides: Any
) -> TrainConfig:
    hyperparameters = Hyperparameters(
        batch_size=8,
        max_epochs=100,
        patience=20,
        learning_rate=1e-3,
        adam_epsilon=1e-4,
        learning_rate_factor=0.1,
        learning_rate_period=2,
        minimum_learning_rate=1e-6,
        weight_decay=0.0,
        minimum_improvement=0.0,
        use_amp=False,
    )
    values: dict[str, Any] = {
        "device": "cpu",
        "detector_candidate_threshold": 0.5,
        "detector_augmentation_factor": 10,
        "discriminator_augmentation_factor": 5,
        "num_workers": 0,
        "pin_memory": False,
        "detector_hyperparameters": hyperparameters,
        "teacher_hyperparameters": hyperparameters,
        "student_hyperparameters": hyperparameters,
    }
    values.update(overrides)
    return TrainConfig(
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        **values,
    )


def _train_manifest_for_config(
    config: TrainConfig, split_manifest_fingerprint: str
) -> TrainManifest:
    return TrainManifest(
        status=ManifestStatus.RUNNING,
        dataset_dir=str(config.dataset_dir.resolve()),
        split_manifest_fingerprint=split_manifest_fingerprint,
        device=config.device,
        seed=config.seed,
        detector_candidate_threshold=config.detector_candidate_threshold,
        detector_augmentation_factor=config.detector_augmentation_factor,
        discriminator_augmentation_factor=config.discriminator_augmentation_factor,
        validation_augmentation_factor=1,
        detector_patch_size=48,
        discriminator_patch_size=24,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _write_raw_dataset_manifest(
    tmp_path: Path,
    dataset_dir: Path,
    *,
    status: ManifestStatus = ManifestStatus.COMPLETE,
    mask_path: str | None,
) -> RawDatasetManifest:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    volume_path = tmp_path / f"{dataset_dir.name}-volume.nii.gz"
    volume_path.touch()
    manifest = RawDatasetManifest(
        status=status,
        sources=[
            RawSource(
                input_dir=str(tmp_path),
                volume_pattern="{subject_id}.nii.gz",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=[
            RawSubject(
                subject_id="subject-1",
                source_id="source",
                volume_path=str(volume_path),
                mask_path=mask_path,
            )
        ],
    )
    manifest.write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    return manifest


def test_train_config_rejects_invalid_and_unavailable_devices(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    with pytest.raises(ValidationError, match="invalid device"):
        _train_config(dataset_dir, tmp_path / "experiment", device="invalid")

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValidationError, match="CUDA device is not available"):
        _train_config(dataset_dir, tmp_path / "experiment", device="cuda")


def test_split_config_requires_split_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="train_size"):
        SplitConfig.model_validate(
            {
                "dataset_dir": _training_dataset(tmp_path),
                "experiment_dir": tmp_path / "experiment",
            }
        )


def test_train_config_requires_training_recipe(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])

    with pytest.raises(ValidationError, match="detector_candidate_threshold"):
        TrainConfig.model_validate(
            {
                "dataset_dir": dataset_dir,
                "experiment_dir": tmp_path / "experiment",
            }
        )


def test_train_config_resume_requires_existing_stage_state(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])

    with pytest.raises(ValidationError, match="training manifest does not exist"):
        _train_config(dataset_dir, tmp_path / "experiment", resume=True)


def test_train_config_resume_rejects_running_stage_without_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).write(layout.train_manifest_path())
    TrainStageManifest(
        status=ManifestStatus.RUNNING,
        stage=DETECTOR_STAGE,
        hyperparameters=config.detector_hyperparameters,
        history=[],
    ).write(layout.stage_manifest_path(DETECTOR_STAGE))

    with pytest.raises(ValidationError, match="no training stage checkpoint"):
        TrainConfig(**config.model_dump(exclude={"resume"}), resume=True)


def test_ensure_training_resume_state_returns_for_existing_latest_checkpoint(
    tmp_path: Path,
) -> None:
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    latest_checkpoint = layout.latest_checkpoint_path(DETECTOR_STAGE)
    latest_checkpoint.parent.mkdir(parents=True)
    latest_checkpoint.touch()

    ensure_training_resume_state(layout)


def test_train_config_resume_accepts_changed_training_settings(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).write(layout.train_manifest_path())
    changed_config = config.model_dump(exclude={"resume"})
    changed_config["detector_hyperparameters"] = (
        config.detector_hyperparameters.model_copy(
            update={"max_epochs": config.detector_hyperparameters.max_epochs + 1}
        )
    )
    TrainStageManifest(
        status=ManifestStatus.COMPLETE,
        stage=DETECTOR_STAGE,
        hyperparameters=config.detector_hyperparameters,
        history=[],
    ).write(layout.stage_manifest_path(DETECTOR_STAGE))

    resumed = TrainConfig(**changed_config, resume=True)

    assert resumed.resume is True


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
def test_split_config_rejects_invalid_split_values(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate(
            {
                "dataset_dir": _training_dataset(tmp_path),
                "experiment_dir": tmp_path / "experiment",
                **overrides,
            }
        )


def test_train_config_accepts_stage_training_settings_and_pin_memory(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    config = TrainConfig.model_validate(
        {
            "dataset_dir": dataset_dir,
            "experiment_dir": tmp_path / "experiment",
            "device": "cpu",
            "detector_candidate_threshold": 0.5,
            "detector_augmentation_factor": 10,
            "discriminator_augmentation_factor": 5,
            "num_workers": 0,
            "pin_memory": True,
            "detector_hyperparameters": Hyperparameters(
                batch_size=4,
                max_epochs=12,
                patience=20,
                learning_rate=1e-3,
                adam_epsilon=1e-4,
                learning_rate_factor=0.1,
                learning_rate_period=2,
                minimum_learning_rate=1e-6,
                weight_decay=0.0,
                minimum_improvement=0.0,
                use_amp=False,
            ),
            "teacher_hyperparameters": Hyperparameters(
                batch_size=5,
                max_epochs=13,
                patience=20,
                learning_rate=1e-3,
                adam_epsilon=1e-4,
                learning_rate_factor=0.1,
                learning_rate_period=2,
                minimum_learning_rate=1e-6,
                weight_decay=0.0,
                minimum_improvement=0.0,
                use_amp=False,
            ),
            "student_hyperparameters": Hyperparameters(
                batch_size=6,
                max_epochs=14,
                patience=20,
                learning_rate=1e-3,
                adam_epsilon=1e-4,
                learning_rate_factor=0.1,
                learning_rate_period=2,
                minimum_learning_rate=1e-6,
                weight_decay=0.0,
                minimum_improvement=0.0,
                use_amp=False,
            ),
        }
    )

    assert config.pin_memory is True
    assert config.detector_hyperparameters.batch_size == 4
    assert config.detector_hyperparameters.max_epochs == 12
    assert config.teacher_hyperparameters.batch_size == 5
    assert config.teacher_hyperparameters.max_epochs == 13
    assert config.student_hyperparameters.batch_size == 6
    assert config.student_hyperparameters.max_epochs == 14


def test_split_config_requires_split_sizes_to_sum_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        SplitConfig(
            dataset_dir=_training_dataset(tmp_path),
            experiment_dir=tmp_path / "experiment",
            train_size=0.6,
            validation_size=0.1,
            test_size=0.1,
        )


def test_split_config_requires_preprocessed_manifest(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(ValidationError, match="preprocessed manifest does not exist"):
        SplitConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            train_size=0.7,
            validation_size=0.1,
            test_size=0.2,
        )


def test_split_config_rejects_non_complete_preprocessed_manifest(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest.read(manifest_path).model_copy(
        update={"status": ManifestStatus.RUNNING}
    ).write(manifest_path)

    with pytest.raises(ValidationError, match="has status 'running'"):
        SplitConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            train_size=0.7,
            validation_size=0.1,
            test_size=0.2,
        )


def test_split_config_rejects_subjects_with_missing_masks(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    manifest = PreprocessedDatasetManifest.read(manifest_path)
    maskless_subject = PreprocessedSubject(
        subject_id="subject-maskless",
        original_volume_path=str(tmp_path / "maskless-volume.nii.gz"),
        bounding_box=((0, 1), (0, 1), (0, 1)),
        variants=[
            PreprocessedVariant(
                volume_path=str(tmp_path / "maskless-volume.nii.gz"),
                mask_path=None,
                frst_path=str(tmp_path / "maskless-frst.nii.gz"),
            )
        ],
    )
    manifest.model_copy(
        update={"subjects": [*manifest.subjects, maskless_subject]}
    ).write(
        manifest_path
    )

    with pytest.raises(
        ValidationError, match="subjects missing masks: subject-maskless"
    ):
        SplitConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            train_size=0.7,
            validation_size=0.1,
            test_size=0.2,
        )


def test_split_config_requires_dataset_dir(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="dataset_dir does not exist"):
        SplitConfig(
            dataset_dir=tmp_path / "missing",
            experiment_dir=tmp_path / "experiment",
            train_size=0.7,
            validation_size=0.1,
            test_size=0.2,
        )


def test_train_config_requires_dataset_and_preprocessed_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="dataset_dir does not exist"):
        _train_config(tmp_path / "missing", tmp_path / "experiment")


def test_train_config_rejects_non_complete_preprocessed_manifest(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest.read(manifest_path).model_copy(
        update={"status": ManifestStatus.RUNNING}
    ).write(manifest_path)

    with pytest.raises(ValidationError, match="status 'running'"):
        _train_config(dataset_dir, tmp_path / "experiment")


def test_train_config_requires_split_manifest(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)

    with pytest.raises(ValidationError, match="split manifest does not exist"):
        _train_config(dataset_dir, tmp_path / "experiment")


def test_train_config_resume_rejects_complete_training_manifest(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).model_copy(update={"status": ManifestStatus.COMPLETE}).write(
        layout.train_manifest_path()
    )

    with pytest.raises(ValidationError, match="training run is already complete"):
        _train_config(dataset_dir, tmp_path / "experiment", resume=True)


def test_train_config_resume_accepts_complete_stage_manifest(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).write(layout.train_manifest_path())
    TrainStageManifest(
        status=ManifestStatus.COMPLETE,
        stage=DETECTOR_STAGE,
        hyperparameters=config.detector_hyperparameters,
        history=[],
    ).write(layout.stage_manifest_path(DETECTOR_STAGE))

    resumed = _train_config(dataset_dir, tmp_path / "experiment", resume=True)

    assert resumed.resume is True


def test_train_config_rejects_split_manifest_for_another_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    experiment_dir = tmp_path / "experiment"
    SplitManifest(
        status=ManifestStatus.COMPLETE,
        created_at=timestamp(),
        updated_at=timestamp(),
        dataset_dir=str((tmp_path / "other-dataset").resolve()),
        preprocessed_manifest_fingerprint="different-dataset",
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        train_subject_ids=["subject-0"],
        validation_subject_ids=["subject-1"],
        test_subject_ids=["subject-2"],
    ).write(ExperimentLayout(experiment_dir=experiment_dir).split_manifest_path())

    with pytest.raises(ValidationError, match="does not match"):
        _train_config(dataset_dir, experiment_dir)

    dataset_dir = tmp_path / "empty-dataset"
    dataset_dir.mkdir()
    with pytest.raises(ValidationError, match="preprocessed manifest does not exist"):
        _train_config(dataset_dir, tmp_path / "experiment")


def test_preprocess_config_requires_readable_raw_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="raw manifest does not exist"):
        PreprocessConfig(dataset_dir=tmp_path, augmentation_factor=1)

    manifest_path = DatasetLayout(dataset_dir=tmp_path).raw_manifest_path()
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid manifest at"):
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
        raw_manifest_fingerprint="test-raw-manifest",
    ).write(manifest_path)
    _complete_split(tmp_path, [])
    _train_config(dataset_dir, tmp_path / "experiment")

    missing_subjects = [
        PreprocessedSubject(
            subject_id=f"subject-{index}",
            original_volume_path=str(tmp_path / f"missing-volume-{index}"),
            bounding_box=((0, 1), (0, 1), (0, 1)),
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
        raw_manifest_fingerprint="test-raw-manifest",
    ).write(manifest_path)
    with pytest.raises(
        ValidationError,
        match="different preprocessed manifest",
    ):
        _train_config(dataset_dir, tmp_path / "experiment")


def test_infer_config_accepts_raw_dataset_explicit_input(tmp_path: Path) -> None:
    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    detector_checkpoint.touch()
    student_checkpoint.touch()
    dataset_dir = tmp_path / "dataset"
    volume_path = tmp_path / "raw.nii.gz"
    volume_path.touch()
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[
            RawSource(
                input_dir=str(tmp_path),
                volume_pattern="{subject_id}.nii.gz",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=[
            RawSubject(
                subject_id="subject-1",
                source_id="source",
                volume_path=str(volume_path),
            )
        ],
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())

    config = InferConfig(
        output_dir=tmp_path / "inference",
        dataset_dir=dataset_dir,
        detector_checkpoint_path=detector_checkpoint,
        student_checkpoint_path=student_checkpoint,
        device="cpu",
    )

    assert config.dataset_dir == dataset_dir
    assert config.model_dump(mode="json")["dataset_dir"] == str(dataset_dir)


def test_infer_config_rejects_missing_explicit_detector_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=None,
    )
    student_checkpoint = tmp_path / "student.pth"
    student_checkpoint.touch()

    with pytest.raises(ValidationError, match="detector checkpoint does not exist"):
        InferConfig(
            output_dir=tmp_path / "inference",
            dataset_dir=dataset_dir,
            detector_checkpoint_path=tmp_path / "missing-detector.pth",
            student_checkpoint_path=student_checkpoint,
            device="cpu",
        )


def test_infer_config_requires_device_and_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        InferConfig.model_validate({"device": "cpu"})


def test_evaluate_config_accepts_explicit_checkpoints(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    raw_manifest_path = DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    raw_subjects = [
        RawSubject(
            subject_id="subject-1",
            source_id="source",
            volume_path=str(tmp_path / "volume.nii.gz"),
            mask_path=str(tmp_path / "mask.nii.gz"),
        )
    ]
    (tmp_path / "volume.nii.gz").touch()
    (tmp_path / "mask.nii.gz").touch()
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[
            RawSource(
                input_dir=str(tmp_path),
                volume_pattern="{subject_id}.nii.gz",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=raw_subjects,
    ).write(raw_manifest_path)
    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    detector_checkpoint.touch()
    student_checkpoint.touch()

    config = EvaluateConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "evaluation",
        detector_checkpoint_path=detector_checkpoint,
        student_checkpoint_path=student_checkpoint,
        device="cpu",
    )

    assert config.output_dir == tmp_path / "evaluation"


def test_evaluate_config_derives_paths_from_experiment(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    experiment_dir = tmp_path / "experiment"
    layout = ExperimentLayout(experiment_dir=experiment_dir)
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    train_config = _train_config(dataset_dir, experiment_dir)
    _train_manifest_for_config(
        train_config, content_fingerprint(split_manifest)
    ).model_copy(update={"status": ManifestStatus.COMPLETE}).write(
        layout.train_manifest_path()
    )
    detector_checkpoint = layout.best_checkpoint_path(DETECTOR_STAGE)
    student_checkpoint = layout.best_checkpoint_path(STUDENT_STAGE)
    detector_checkpoint.parent.mkdir(parents=True)
    student_checkpoint.parent.mkdir(parents=True)
    detector_checkpoint.touch()
    student_checkpoint.touch()

    config = EvaluateConfig(
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        device="cpu",
    )

    assert config.output_dir == experiment_dir
    assert config.detector_checkpoint_path == detector_checkpoint
    assert config.student_checkpoint_path == student_checkpoint


def test_evaluate_config_rejects_non_complete_experiment_manifest(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).write(layout.train_manifest_path())

    with pytest.raises(ValidationError, match="status 'running'"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            device="cpu",
        )


def test_evaluate_config_rejects_mismatched_preprocessed_manifest(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    split_manifest = SplitManifest.read(layout.split_manifest_path())
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, content_fingerprint(split_manifest)
    ).model_copy(update={"status": ManifestStatus.COMPLETE}).write(
        layout.train_manifest_path()
    )
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest.read(manifest_path).model_copy(
        update={"subjects": []}
    ).write(manifest_path)

    with pytest.raises(ValidationError, match="different preprocessed manifest"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            device="cpu",
        )


def test_evaluate_config_rejects_mismatched_train_manifest_split(
    tmp_path: Path,
) -> None:
    dataset_dir = _training_dataset(tmp_path)
    _complete_split(tmp_path, [f"subject-{index}" for index in range(7)])
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    config = _train_config(dataset_dir, tmp_path / "experiment")
    _train_manifest_for_config(
        config, "different-split-fingerprint"
    ).model_copy(update={"status": ManifestStatus.COMPLETE}).write(
        layout.train_manifest_path()
    )

    with pytest.raises(ValidationError, match="different split manifest"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            experiment_dir=tmp_path / "experiment",
            device="cpu",
        )


def test_evaluate_config_requires_complete_explicit_mode(tmp_path: Path) -> None:
    dataset_dir = _training_dataset(tmp_path)
    detector_checkpoint = tmp_path / "detector.pth"
    detector_checkpoint.touch()

    with pytest.raises(ValidationError, match="student_checkpoint_path"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "evaluation",
            detector_checkpoint_path=detector_checkpoint,
            device="cpu",
        )


def test_evaluate_config_rejects_missing_explicit_detector_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    mask_path = tmp_path / "mask.nii.gz"
    mask_path.touch()
    _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=str(mask_path),
    )
    student_checkpoint = tmp_path / "student.pth"
    student_checkpoint.touch()

    with pytest.raises(ValidationError, match="detector checkpoint does not exist"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "evaluation",
            detector_checkpoint_path=tmp_path / "missing-detector.pth",
            student_checkpoint_path=student_checkpoint,
            device="cpu",
        )


def test_evaluate_config_rejects_maskless_raw_subjects(tmp_path: Path) -> None:
    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    detector_checkpoint.touch()
    student_checkpoint.touch()
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset_manifest(
        tmp_path,
        dataset_dir,
        mask_path=None,
    )

    with pytest.raises(ValidationError, match="cannot evaluate subjects without masks"):
        EvaluateConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "evaluation",
            detector_checkpoint_path=detector_checkpoint,
            student_checkpoint_path=student_checkpoint,
            device="cpu",
        )


def test_evaluate_config_requires_explicit_values_without_experiment(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        EvaluateConfig(dataset_dir=tmp_path, device="cpu")


def test_target_centered_config_validates_threshold(tmp_path: Path) -> None:
    values = {
        "experiment_layout": ExperimentLayout(experiment_dir=tmp_path),
        "stage": "student",
        "split": "train",
        "subjects": [
            PreprocessedSubject(
                subject_id="subject",
                original_volume_path="volume.nii.gz",
                bounding_box=((0, 1), (0, 1), (0, 1)),
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
                original_volume_path="volume.nii.gz",
                bounding_box=((0, 1), (0, 1), (0, 1)),
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
