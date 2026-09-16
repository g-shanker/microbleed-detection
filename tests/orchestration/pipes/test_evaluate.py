"""Tests for held-out component scoring."""

from types import SimpleNamespace

import numpy as np

from microbleednet.core.common.metrics import aggregate_metrics, score_masks
from microbleednet.orchestration.configs import EvaluateConfig
from microbleednet.orchestration.layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    EvaluateManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)
from microbleednet.orchestration.pipes import evaluate


def test_score_masks_matches_overlapping_components() -> None:
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1, 1, 1] = 1
    prediction[3, 3, 3] = 1
    reference[1, 1, 1] = 1
    reference[2, 2, 4] = 1

    metrics = score_masks(prediction, reference)

    assert metrics.model_dump(mode="json") == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "cluster_tpr": 0.5,
        "cluster_precision": 0.5,
    }


def test_score_masks_handles_empty_components() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)

    metrics = score_masks(mask, mask)

    assert metrics.true_positive == 0
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0


def test_score_masks_ignores_background_and_unmatched_pairs() -> None:
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[0, 0, 0] = 1
    reference[2, 2, 2] = 1

    metrics = score_masks(prediction, reference)

    assert metrics.true_positive == 0
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1


def test_aggregate_metrics_counts_subjects() -> None:
    metrics = score_masks(
        np.ones((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2, 2), dtype=np.uint8),
    )

    aggregate = aggregate_metrics([metrics])

    assert aggregate.false_positives_per_subject == 1.0


def test_execute_writes_held_out_evaluation_manifest(tmp_path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiment"
    dataset_dir = tmp_path / "dataset"
    subject = PreprocessedSubject(
        subject_id="subject-1",
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path="reference", frst_path="frst"
            )
        ],
    )
    experiment_layout = ExperimentLayout(experiment_dir=experiment_dir)
    for stage in (DETECTOR_STAGE, STUDENT_STAGE):
        checkpoint = experiment_layout.best_checkpoint_path(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1, 1, 1] = 1
    reference[1, 1, 1] = 1

    monkeypatch.setattr(
        evaluate.TrainManifest,
        "read",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        evaluate.SplitManifest,
        "read",
        lambda _: SimpleNamespace(test_subject_ids=["subject-1"]),
    )
    monkeypatch.setattr(
        evaluate.PreprocessedDatasetManifest,
        "read",
        lambda _: SimpleNamespace(subjects=[subject]),
    )
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(
            subjects=[SimpleNamespace(subject_id="subject-1", output_path="prediction")]
        ),
    )
    monkeypatch.setattr(evaluate.infer, "execute", lambda _: None)
    monkeypatch.setattr(evaluate.core_io, "load_volume", lambda path: path)
    monkeypatch.setattr(
        evaluate.core_io,
        "nifti_to_numpy",
        lambda volume: prediction if volume == "prediction" else reference,
    )

    evaluate.execute(
        EvaluateConfig(experiment_dir=experiment_dir, dataset_dir=dataset_dir)
    )

    manifest = EvaluateManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).evaluation_manifest_path()
    )
    assert [item.subject_id for item in manifest.subjects] == ["subject-1"]
    assert manifest.subjects[0].metrics.true_positive == 1
    assert manifest.aggregate.true_positive == 1