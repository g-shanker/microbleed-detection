from pathlib import Path

import pytest
from pydantic import ValidationError

from microbleednet.core.datamodels import TrainingSettings
from microbleednet.core import io as core_io
from microbleednet.orchestration.manifests import (
    InferManifest,
    Manifest,
    ManifestStatus,
    RawDatasetManifest,
    TrainManifest,
    timestamp,
)


def _envelope(**overrides: object) -> dict:
    now = timestamp()
    base = {
        "schema_version": 1,
        "status": ManifestStatus.COMPLETE.value,
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)
    return base


def test_failed_manifest_requires_error_message() -> None:
    with pytest.raises(ValidationError, match="nonempty error string"):
        Manifest.model_validate(_envelope(status=ManifestStatus.FAILED.value))


def test_non_failed_manifest_rejects_error_message() -> None:
    with pytest.raises(ValidationError, match="only a failed manifest"):
        Manifest.model_validate(_envelope(error="boom"))


def test_raw_dataset_manifest_rejects_duplicate_subject_ids() -> None:
    subject = {
        "subject_id": "s1",
        "source_id": "source",
        "volume_path": "/a",
        "mask_path": "/a-mask",
    }
    with pytest.raises(ValidationError, match="duplicate subject ID"):
        RawDatasetManifest.model_validate(
            _envelope(
                manifest_type="raw_dataset", sources=[], subjects=[subject, subject]
            )
        )


def test_raw_dataset_manifest_rejects_duplicate_source_ids() -> None:
    source = {
        "input_dir": "/input",
        "mask_dir": "/mask",
        "volume_pattern": "{subject_id}.nii.gz",
        "mask_pattern": "{subject_id}.nii.gz",
        "source_id": "source",
        "modality": "QSM",
        "added_on": timestamp(),
    }
    with pytest.raises(ValidationError, match="duplicate source ID"):
        RawDatasetManifest.model_validate(
            _envelope(
                manifest_type="raw_dataset", sources=[source, source], subjects=[]
            )
        )


def test_read_manifest_rejects_unversioned_payload(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    core_io.write_json_atomic(path, {"status": "complete"})
    with pytest.raises(ValueError, match="not a versioned manifest"):
        RawDatasetManifest.read(path)


def test_read_manifest_rejects_incomplete_status(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    core_io.write_json_atomic(
        path,
        _envelope(
            status=ManifestStatus.RUNNING.value,
            manifest_type="raw_dataset",
            sources=[],
            subjects=[],
        ),
    )
    with pytest.raises(ValueError, match="a consumer may only read a complete"):
        RawDatasetManifest.read(path)


def test_read_manifest_rejects_payload_that_fails_schema_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.json"
    # schema_version is present, but required manifest fields are missing.
    core_io.write_json_atomic(path, _envelope())
    with pytest.raises(ValueError, match="invalid manifest at"):
        RawDatasetManifest.read(path)


def test_train_manifest_round_trips_split_and_training_settings(
    tmp_path: Path,
) -> None:
    now = timestamp()
    manifest = TrainManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        dataset_dir="C:/datasets/preprocessed",
        device="cuda:0",
        train_size=0.7,
        train_subject_ids=["train-1"],
        validation_subject_ids=["validation-1"],
        detector_candidate_threshold=0.5,
        detector_augmentation_factor=10,
        discriminator_augmentation_factor=5,
        validation_augmentation_factor=1,
        detector_patch_size=48,
        discriminator_patch_size=24,
        num_workers=0,
        pin_memory=False,
        training_settings=TrainingSettings(),
        detector_history=[],
        teacher_history=[],
        student_history=[],
    )
    path = tmp_path / "train.json"

    manifest.write(path)

    loaded = TrainManifest.read(path)
    assert loaded.dataset_dir == "C:/datasets/preprocessed"
    assert loaded.device == "cuda:0"
    assert loaded.train_subject_ids == ["train-1"]
    assert loaded.validation_subject_ids == ["validation-1"]
    assert loaded.training_settings.batch_size == 8
    assert loaded.detector_history == []


def test_inference_manifest_round_trips_recipe_and_results(tmp_path: Path) -> None:
    now = timestamp()
    manifest = InferManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        device="cpu",
        detector_checkpoint_path="C:/experiments/train/detector/best_model.pth",
        student_checkpoint_path="C:/experiments/train/student/best_model.pth",
        detector_threshold=0.5,
        student_threshold=0.5,
        discriminator_patch_size=24,
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
        subjects=[
            {
                "subject_id": "subject-1",
                "output_path": "C:/experiments/infer/subject-1/detections.nii.gz",
            }
        ],
    )
    path = tmp_path / "infer.json"

    manifest.write(path)

    loaded = InferManifest.read(path)
    assert loaded.manifest_type == "inference"
    assert loaded.subjects[0].subject_id == "subject-1"
    assert loaded.subjects[0].output_path.endswith("detections.nii.gz")
    assert loaded.discriminator_patch_size == 24
