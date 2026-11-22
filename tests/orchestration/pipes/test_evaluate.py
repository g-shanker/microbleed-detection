"""Tests for held-out component scoring and the evaluate pipe."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from microbleednet.core.common.metrics import aggregate_metrics, score_masks
from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import EvaluateConfig, InferConfig
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import (
    EvaluateManifest,
    ManifestStatus,
    content_fingerprint,
)
from microbleednet.orchestration.pipes import evaluate
from tests.support import (
    raw_subjects,
    save_nifti,
    train_config,
    train_manifest_for,
    write_raw_manifest,
)
from tests.support_pipes import (
    write_experiment_inputs,
    write_explicit_inputs,
    write_inference_manifest,
)


def volume(
    shape: tuple[int, ...], *regions: Any, value: float = 1, dtype: Any = np.uint8
) -> np.ndarray:
    """Build a zero volume with ``value`` written at each index ``region``."""
    array = np.zeros(shape, dtype=dtype)
    for region in regions:
        array[region] = value
    return array


# score_masks / aggregate_metrics


def test_score_masks_matches_overlapping_components() -> None:
    prediction = volume((5, 5, 5), (1, 1, 1), (3, 3, 3))
    reference = volume((5, 5, 5), (1, 1, 1), (2, 2, 4))

    metrics = score_masks(prediction, reference)

    assert metrics.model_dump(mode="json") == {
        "detected_reference_clusters": 1,
        "overlapping_prediction_clusters": 1,
        "false_positive": 1,
        "false_negative": 1,
        "cluster_tpr": 0.5,
        "cluster_precision": 0.5,
        "dice_score": 0.5,
    }


def test_score_masks_handles_empty_components() -> None:
    mask = volume((3, 3, 3))

    assert score_masks(mask, mask).model_dump(mode="json") == {
        "detected_reference_clusters": 0,
        "overlapping_prediction_clusters": 0,
        "false_positive": 0,
        "false_negative": 0,
        "cluster_tpr": 0.0,
        "cluster_precision": 0.0,
        "dice_score": 1.0,
    }


@pytest.mark.parametrize(
    ("prediction_values", "reference_values", "expected"),
    [
        ([1, 1, 0], [1, 1, 0], 1.0),
        ([1, 1, 0], [0, 1, 0], 2 / 3),
        ([1, 0, 0], [0, 0, 1], 0.0),
        ([0, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 0, 0], 0.0),
        ([255, 255, 0], [1, 1, 0], 1.0),
    ],
)
def test_score_masks_foreground_dice(
    prediction_values: list[int], reference_values: list[int], expected: float
) -> None:
    prediction = np.array(prediction_values, dtype=np.uint8).reshape(3, 1, 1)
    reference = np.array(reference_values, dtype=np.uint8).reshape(3, 1, 1)

    assert score_masks(prediction, reference).dice_score == pytest.approx(expected)


def test_score_masks_ignores_background_and_unmatched_pairs() -> None:
    metrics = score_masks(volume((3, 3, 3), (0, 0, 0)), volume((3, 3, 3), (2, 2, 2)))

    assert metrics.detected_reference_clusters == 0
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1


@pytest.mark.parametrize("merged_prediction", [False, True])
def test_score_masks_counts_all_overlapping_clusters(merged_prediction: bool) -> None:
    connected = volume((7, 7, 7), (slice(1, 6), 3, 3))
    separated = volume((7, 7, 7), (1, 3, 3), (5, 3, 3))
    prediction, reference = (
        (connected, separated) if merged_prediction else (separated, connected)
    )

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == (2 if merged_prediction else 1)
    assert metrics.overlapping_prediction_clusters == (1 if merged_prediction else 2)
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.cluster_tpr == 1.0
    assert metrics.cluster_precision == 1.0


def test_score_masks_does_not_trade_lesion_hits_for_overlap_volume() -> None:
    shape = (14, 8, 3)
    prediction = volume(shape, (slice(1, 12), 1, 1), (1, 5, 1))
    reference = volume(shape, (slice(1, 10), 1, 1), (1, slice(1, 6), 1), (11, 1, 1))

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 2
    assert metrics.overlapping_prediction_clusters == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0


def test_score_masks_binarizes_labels_and_uses_26_connectivity() -> None:
    reference = volume((4, 4, 4), (1, 1, 1), dtype=float)
    reference[2, 2, 2] = 255
    prediction = (reference > 0).astype(np.uint8)

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 1
    assert metrics.overlapping_prediction_clusters == 1
    assert metrics.false_negative == 0
    assert metrics.dice_score == 1.0
    assert score_masks(reference, prediction) == metrics


def test_score_masks_accepts_one_voxel_overlap_without_dice_threshold() -> None:
    shape = (20, 20, 20)
    prediction = volume(shape, (slice(1, 15),) * 3)
    reference = volume(shape, (slice(14, 18),) * 3)

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.dice_score < 0.001


def test_aggregate_metrics_averages_subject_dice() -> None:
    perfect = score_masks(np.ones((3, 3, 3)), np.ones((3, 3, 3)))
    missed = score_masks(np.zeros((1, 1, 1)), np.ones((1, 1, 1)))

    assert aggregate_metrics([perfect, missed]).dice_score == 0.5


def test_aggregate_metrics_of_no_subjects_is_all_zero() -> None:
    assert aggregate_metrics([]).model_dump(mode="json") == {
        "detected_reference_clusters": 0,
        "overlapping_prediction_clusters": 0,
        "false_positive": 0,
        "false_negative": 0,
        "cluster_tpr": 0.0,
        "cluster_precision": 0.0,
        "dice_score": 0.0,
        "false_positives_per_subject": 0.0,
    }


def test_aggregate_metrics_pools_reference_and_prediction_hits_separately() -> None:
    connected = volume((7, 7, 7), (slice(1, 6), 3, 3))
    separated = volume((7, 7, 7), (1, 3, 3), (5, 3, 3))
    split = score_masks(separated, connected)
    missed = score_masks(np.zeros((3, 3, 3)), np.ones((3, 3, 3)))
    spurious = score_masks(np.ones((3, 3, 3)), np.zeros((3, 3, 3)))

    metrics = aggregate_metrics([split, missed, spurious])

    assert metrics.detected_reference_clusters == 1
    assert metrics.overlapping_prediction_clusters == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.cluster_tpr == 0.5
    assert metrics.cluster_precision == pytest.approx(2 / 3)
    assert metrics.false_positives_per_subject == pytest.approx(1 / 3)


# evaluate_subjects


@pytest.fixture
def layout(tmp_path: Path) -> ExperimentLayout:
    return ExperimentLayout(experiment_dir=tmp_path / "experiment")


def evaluate_subjects(
    layout: ExperimentLayout, dataset_dir: Path, subjects: list[tuple[str, str | None]]
) -> None:
    config = EvaluateConfig.model_construct(device="cpu")
    evaluate.evaluate_subjects(config, dataset_dir, layout, subjects)


def test_evaluate_subjects_writes_overlap_metrics_from_nifti(
    tmp_path: Path, layout: ExperimentLayout
) -> None:
    prediction = volume((7, 7, 7), (1, 3, 3), (5, 3, 3))
    reference = volume((7, 7, 7), (slice(1, 6), 3, 3), value=255)
    write_inference_manifest(layout, tmp_path, {"subject-1": prediction})
    reference_path = save_nifti(tmp_path / "reference.nii.gz", data=reference)

    evaluate_subjects(layout, tmp_path, [("subject-1", reference_path)])

    manifest = EvaluateManifest.read(layout.evaluation_manifest_path())
    expected = {
        "detected_reference_clusters": 1,
        "overlapping_prediction_clusters": 2,
        "false_positive": 0,
        "false_negative": 0,
        "cluster_tpr": 1.0,
        "cluster_precision": 1.0,
    }
    assert manifest.subjects[0].metrics.model_dump(include=set(expected)) == expected
    assert manifest.aggregate.model_dump(include=set(expected)) == expected


def test_evaluate_subjects_records_provenance(
    tmp_path: Path, layout: ExperimentLayout
) -> None:
    mask = volume((3, 3, 3), (1, 1, 1))
    write_inference_manifest(layout, tmp_path, {"subject-1": mask})
    reference_path = save_nifti(tmp_path / "reference.nii.gz", data=mask)

    evaluate_subjects(layout, tmp_path, [("subject-1", reference_path)])

    manifest = EvaluateManifest.read(layout.evaluation_manifest_path())
    assert manifest.status is ManifestStatus.COMPLETE
    assert manifest.dataset_dir == str(tmp_path.resolve())
    assert manifest.device == "cpu"
    assert manifest.inference_manifest_path == str(
        layout.inference_manifest_path().resolve()
    )
    assert [subject.subject_id for subject in manifest.subjects] == ["subject-1"]


@pytest.mark.parametrize(
    ("status", "predicted", "reference", "summary", "fix"),
    [
        (
            ManifestStatus.RUNNING,
            [],
            "reference",
            "Cannot evaluate incomplete inference output",
            "Rerun the evaluate command",
        ),
        (
            ManifestStatus.COMPLETE,
            ["subject-1"],
            None,
            "Evaluation requires a reference mask",
            "Provide reference masks",
        ),
        (
            ManifestStatus.COMPLETE,
            [],
            "reference",
            "Inference output is missing",
            "Rerun the evaluate command",
        ),
    ],
    ids=["incomplete-inference", "no-reference-mask", "missing-inference-output"],
)
def test_evaluate_subjects_rejects_unscorable_input(
    tmp_path: Path,
    layout: ExperimentLayout,
    status: ManifestStatus,
    predicted: list[str],
    reference: str | None,
    summary: str,
    fix: str,
) -> None:
    write_inference_manifest(
        layout, tmp_path, {subject: volume((3, 3, 3)) for subject in predicted}, status
    )

    with pytest.raises(ApplicationError) as error:
        evaluate_subjects(layout, tmp_path, [("subject-1", reference)])

    assert summary in str(error.value)
    assert fix in str(error.value)


# execute

REFERENCE = volume((3, 3, 3), (1, 1, 1))


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A finished training experiment with a test subject and faked inference."""
    inputs = write_experiment_inputs(tmp_path)
    dataset_dir, experiment_dir = tmp_path / "dataset", tmp_path / "experiment"
    layout = ExperimentLayout(experiment_dir=experiment_dir)
    # Preprocessed masks are empty files, so scoring must read the raw masks.
    write_raw_manifest(
        dataset_dir, raw_subjects(3, directory=tmp_path / "raw", data=REFERENCE)
    )
    train_manifest_for(
        train_config(dataset_dir, experiment_dir),
        content_fingerprint(inputs.split),
        status=ManifestStatus.COMPLETE,
    ).write(layout.train_manifest_path())
    state = SimpleNamespace(
        inputs=inputs,
        layout=layout,
        models=(object(), object()),
        loads=[],
        inferred=[],
        config=EvaluateConfig.model_validate(
            {
                "experiment_dir": experiment_dir,
                "dataset_dir": dataset_dir,
                "device": "cpu",
                "use_amp": True,
                "bias_field_correction": "fast",
            }
        ),
    )

    def infer_subjects(*args: Any) -> None:
        state.inferred.append(args)
        write_inference_manifest(
            layout, tmp_path / "predictions", {s.subject_id: REFERENCE for s in args[4]}
        )

    monkeypatch.setattr(
        evaluate.infer,
        "load_inference_models",
        lambda *args: state.loads.append(args) or state.models,
    )
    monkeypatch.setattr(evaluate.infer, "infer_subjects", infer_subjects)
    monkeypatch.setattr(evaluate, "release_gpu_memory", lambda: None)
    return state


def test_experiment_mode_infers_test_subjects_with_experiment_checkpoints(
    experiment: SimpleNamespace,
) -> None:
    config, layout = experiment.config, experiment.layout

    evaluate.execute(config)

    detector = layout.best_checkpoint_path("detector")
    student = layout.best_checkpoint_path("student")
    assert experiment.loads == [("cpu", detector, student)]
    assert experiment.inferred == [
        (
            "cpu",
            layout,
            detector,
            student,
            experiment.inputs.subjects[-1:],
            *experiment.models,
            True,
        )
    ]


def test_experiment_mode_scores_test_subjects_against_raw_masks(
    experiment: SimpleNamespace,
) -> None:
    evaluate.execute(experiment.config)

    manifest = EvaluateManifest.read(experiment.layout.evaluation_manifest_path())
    assert [subject.subject_id for subject in manifest.subjects] == ["subject-2"]
    assert manifest.subjects[0].metrics.dice_score == 1.0
    assert manifest.aggregate.detected_reference_clusters == 1
    assert manifest.aggregate.overlapping_prediction_clusters == 1
    assert manifest.aggregate.dice_score == 1.0


@pytest.fixture
def explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A raw dataset with explicit checkpoints and a faked inference pipe.

    Inference finds subject-0 perfectly and misses subject-1 entirely.
    """
    inputs = write_explicit_inputs(
        tmp_path,
        subjects=raw_subjects(2, directory=tmp_path / "raw", data=REFERENCE),
    )
    state = SimpleNamespace(inputs=inputs, infer_configs=[])

    def infer_execute(config: InferConfig) -> None:
        state.infer_configs.append(config)
        write_inference_manifest(
            ExperimentLayout(experiment_dir=config.output_dir),
            tmp_path / "predictions",
            {"subject-0": REFERENCE, "subject-1": volume((3, 3, 3))},
        )

    monkeypatch.setattr(evaluate.infer, "execute", infer_execute)
    return state


@pytest.mark.parametrize("bias_field_correction", ["fast", "n4"])
def test_explicit_mode_runs_inference_with_matching_config(
    explicit: SimpleNamespace, bias_field_correction: str
) -> None:
    values = {**explicit.inputs, "bias_field_correction": bias_field_correction}

    evaluate.execute(EvaluateConfig.model_validate(values))

    assert explicit.infer_configs == [InferConfig(**values)]


def test_explicit_mode_scores_every_raw_subject(explicit: SimpleNamespace) -> None:
    config = EvaluateConfig.model_validate(explicit.inputs)

    evaluate.execute(config)

    layout = ExperimentLayout(experiment_dir=config.output_dir)
    manifest = EvaluateManifest.read(layout.evaluation_manifest_path())
    assert [
        (subject.subject_id, subject.metrics.dice_score)
        for subject in manifest.subjects
    ] == [("subject-0", 1.0), ("subject-1", 0.0)]
