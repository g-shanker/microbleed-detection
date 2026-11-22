from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from microbleednet.core import io as core_io
from microbleednet.core.datamodels import EvaluationAggregate, EvaluationMetrics
from microbleednet.orchestration.manifests import (
    EvaluatedSubject,
    EvaluateManifest,
    InferManifest,
    InferredSubject,
    Manifest,
    ManifestStatus,
    PatchManifest,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
    TrainManifest,
    TrainStageManifest,
)
from tests.support import hyperparameters, raw_source, raw_subjects

COMPLETE = ManifestStatus.COMPLETE
RUNNING = ManifestStatus.RUNNING
M = TypeVar("M", bound=Manifest)

METRICS = EvaluationMetrics(
    detected_reference_clusters=2,
    overlapping_prediction_clusters=4,
    false_positive=1,
    false_negative=3,
    cluster_tpr=0.4,
    cluster_precision=4 / 5,
    dice_score=0.75,
)
AGGREGATE = EvaluationAggregate(**METRICS.model_dump(), false_positives_per_subject=1.0)


def _round_trip(manifest: M, tmp_path: Path) -> M:
    path = tmp_path / "manifest.json"
    manifest.write(path)
    return type(manifest).read(path)


def _infer_manifest(
    status: ManifestStatus = COMPLETE, output_path: str = ""
) -> InferManifest:
    return InferManifest(
        status=status,
        device="cpu",
        detector_checkpoint_path="detector.pth",
        student_checkpoint_path="student.pth",
        detector_threshold=0.5,
        student_threshold=0.5,
        discriminator_patch_size=24,
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
        subjects=[
            InferredSubject(
                subject_id="subject-1",
                detector_output_path=output_path,
                discriminator_output_path=output_path,
            )
        ],
    )


def _patch_manifest(**overrides: Any) -> PatchManifest:
    values: dict[str, Any] = {
        "status": COMPLETE,
        "stage": "detector",
        "split": "train",
        "subject_ids": ["subject-1"],
        "patch_size": 48,
        "augmentation_factor": 1,
        "use_amp": False,
        "records": [],
    }
    values.update(overrides)
    return PatchManifest(**values)


def _preprocessed_manifest_without_variants() -> PreprocessedDatasetManifest:
    return PreprocessedDatasetManifest(
        status=COMPLETE,
        subjects=[
            PreprocessedSubject(
                subject_id="subject-1",
                original_volume_path="volume",
                brain_mask_path="brain-mask",
                bounding_box=((0, 1), (0, 1), (0, 1)),
                variants=[],
            )
        ],
        augmentation_factor=1,
        bias_field_correction="fast",
        raw_manifest_fingerprint="raw",
    )


def _evaluate_manifest() -> EvaluateManifest:
    return EvaluateManifest(
        status=COMPLETE,
        dataset_dir="C:/datasets/preprocessed",
        device="cpu",
        inference_manifest_path="C:/experiments/manifests/infer.json",
        subjects=[EvaluatedSubject(subject_id="subject-1", metrics=METRICS)],
        aggregate=AGGREGATE,
    )


def test_manifest_owns_lifecycle_timestamps(tmp_path: Path) -> None:
    manifest = Manifest(status=COMPLETE)
    path = tmp_path / "manifest.json"

    manifest.write(path)
    first = core_io.read_json(path)
    manifest.write(path)
    second = core_io.read_json(path)

    assert first["created_at"] == second["created_at"] == manifest.created_at
    assert first["updated_at"] != manifest.updated_at
    assert second["updated_at"] != first["updated_at"]


def test_manifest_write_preserves_creation_timestamp_on_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    first = Manifest(status=COMPLETE)
    first.write(path)

    replacement = Manifest(status=COMPLETE)
    replacement.write(path)
    persisted = core_io.read_json(path)

    assert persisted["created_at"] == first.created_at
    assert persisted["created_at"] != replacement.created_at


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("subjects", raw_subjects(1) * 2, "Duplicate subject ID"),
        ("sources", [raw_source()] * 2, "Duplicate source ID"),
    ],
)
def test_raw_dataset_manifest_rejects_duplicate_ids(
    field: str, values: list, message: str
) -> None:
    fields: dict[str, Any] = {"sources": [], "subjects": [], field: values}

    with pytest.raises(ValidationError, match=message):
        RawDatasetManifest(status=COMPLETE, **fields)


def test_read_manifest_returns_incomplete_status(tmp_path: Path) -> None:
    manifest = RawDatasetManifest(status=RUNNING, sources=[], subjects=[])

    assert _round_trip(manifest, tmp_path).status is RUNNING


def test_read_manifest_rejects_payload_that_fails_schema_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.json"
    Manifest(status=COMPLETE).write(path)

    with pytest.raises(ValueError, match="Manifest schema validation failed"):
        RawDatasetManifest.read(path)


def test_read_manifest_wraps_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read manifest") as raised:
        RawDatasetManifest.read(path)

    assert raised.value.__cause__ is not None


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (_preprocessed_manifest_without_variants, "incomplete variants"),
        (_patch_manifest, "contains no records"),
        (_infer_manifest, "without outputs"),
    ],
    ids=["preprocessed", "patch", "inference"],
)
def test_complete_manifest_rejects_unfinished_subject_output(
    build: Callable[[], Manifest], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        build()


@pytest.mark.parametrize(
    ("build", "status"),
    [
        (lambda: _infer_manifest(RUNNING), RUNNING),
        (lambda: _patch_manifest(subject_ids=[]), COMPLETE),
    ],
    ids=["running-inference", "complete-patch-without-subjects"],
)
def test_manifest_accepts_missing_output_when_none_is_owed(
    build: Callable[[], Manifest], status: ManifestStatus
) -> None:
    assert build().status is status


def test_train_manifest_round_trips_split_and_training_settings(
    tmp_path: Path,
) -> None:
    manifest = TrainManifest(
        status=COMPLETE,
        dataset_dir="C:/datasets/preprocessed",
        split_manifest_fingerprint="split-fingerprint",
        device="cuda:0",
        seed=42,
        detector_candidate_threshold=0.5,
        detector_augmentation_factor=10,
        discriminator_augmentation_factor=5,
        validation_augmentation_factor=1,
        detector_patch_size=48,
        discriminator_patch_size=24,
        num_workers=0,
        pin_memory=False,
    )

    loaded = _round_trip(manifest, tmp_path)

    assert loaded.dataset_dir == "C:/datasets/preprocessed"
    assert loaded.split_manifest_fingerprint == "split-fingerprint"
    assert loaded.device == "cuda:0"
    assert loaded.seed == 42


def test_train_stage_manifest_round_trips_status_and_history(
    tmp_path: Path,
) -> None:
    manifest = TrainStageManifest(
        status=COMPLETE,
        stage="detector",
        hyperparameters=hyperparameters(),
        history=[{"epoch": 1, "training_loss": 2.0, "validation_loss": 1.0}],
    )

    loaded = _round_trip(manifest, tmp_path)

    assert loaded.status is COMPLETE
    assert loaded.stage == "detector"
    assert loaded.hyperparameters == manifest.hyperparameters
    assert loaded.history == manifest.history


def test_inference_manifest_round_trips_recipe_and_results(tmp_path: Path) -> None:
    manifest = _infer_manifest(output_path="C:/experiments/infer/output.nii.gz")

    loaded = _round_trip(manifest, tmp_path)

    assert loaded.manifest_type == "inference"
    assert loaded.subjects == manifest.subjects
    assert loaded.discriminator_patch_size == 24


def test_evaluation_manifest_round_trips_metrics(tmp_path: Path) -> None:
    loaded = _round_trip(_evaluate_manifest(), tmp_path)

    assert loaded.manifest_type == "evaluation"
    assert loaded.subjects == [
        EvaluatedSubject(subject_id="subject-1", metrics=METRICS)
    ]
    assert loaded.aggregate == AGGREGATE


def test_evaluation_manifest_read_rejects_missing_metric(tmp_path: Path) -> None:
    path = tmp_path / "evaluate.json"
    _evaluate_manifest().write(path)
    payload = core_io.read_json(path)
    del payload["subjects"][0]["metrics"]["overlapping_prediction_clusters"]
    del payload["aggregate"]["overlapping_prediction_clusters"]
    core_io.write_json(path, payload)

    with pytest.raises(ValueError, match="overlapping_prediction_clusters"):
        EvaluateManifest.read(path)
