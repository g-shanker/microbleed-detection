"""Tests for held-out component scoring."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nibabel as nib
import numpy as np
import pytest

from microbleednet.constants import DETECTOR_STAGE, STUDENT_STAGE
from microbleednet.core.common.metrics import aggregate_metrics, score_masks
from microbleednet.orchestration import configs
from microbleednet.orchestration.configs import (
    EvaluateConfig,
)
from microbleednet.orchestration.layouts import (
    DatasetLayout,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    EvaluateManifest,
    ManifestStatus,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    timestamp,
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
        "detected_reference_clusters": 1,
        "overlapping_prediction_clusters": 1,
        "false_positive": 1,
        "false_negative": 1,
        "cluster_tpr": 0.5,
        "cluster_precision": 0.5,
        "dice_score": 0.5,
    }


def test_score_masks_handles_empty_components() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)

    metrics = score_masks(mask, mask)

    assert metrics.detected_reference_clusters == 0
    assert metrics.overlapping_prediction_clusters == 0
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.cluster_tpr == 0.0
    assert metrics.cluster_precision == 0.0
    assert metrics.dice_score == 1.0


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
    prediction_values, reference_values, expected
) -> None:
    prediction = np.array(prediction_values, dtype=np.uint8).reshape(3, 1, 1)
    reference = np.array(reference_values, dtype=np.uint8).reshape(3, 1, 1)

    assert score_masks(prediction, reference).dice_score == pytest.approx(expected)


def test_aggregate_metrics_averages_subject_dice() -> None:
    perfect = score_masks(np.ones((3, 3, 3)), np.ones((3, 3, 3)))
    missed = score_masks(np.zeros((1, 1, 1)), np.ones((1, 1, 1)))

    assert aggregate_metrics([perfect, missed]).dice_score == 0.5
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


def test_score_masks_ignores_background_and_unmatched_pairs() -> None:
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[0, 0, 0] = 1
    reference[2, 2, 2] = 1

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 0
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1


def test_aggregate_metrics_counts_subjects() -> None:
    metrics = score_masks(
        np.ones((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2, 2), dtype=np.uint8),
    )

    aggregate = aggregate_metrics([metrics])

    assert aggregate.false_positives_per_subject == 1.0


@pytest.mark.parametrize("merged_prediction", [False, True])
def test_score_masks_counts_all_overlapping_clusters(merged_prediction: bool) -> None:
    connected = np.zeros((7, 7, 7), dtype=np.uint8)
    connected[1:6, 3, 3] = 1
    separated = np.zeros_like(connected)
    separated[1, 3, 3] = 1
    separated[5, 3, 3] = 1
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
    prediction = np.zeros((14, 8, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1:12, 1, 1] = 1
    prediction[1, 5, 1] = 1
    reference[1:10, 1, 1] = 1
    reference[1, 1:6, 1] = 1
    reference[11, 1, 1] = 1

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 2
    assert metrics.overlapping_prediction_clusters == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0


def test_score_masks_binarizes_labels_and_uses_26_connectivity() -> None:
    reference = np.zeros((4, 4, 4))
    reference[1, 1, 1] = 1
    reference[2, 2, 2] = 255
    prediction = (reference > 0).astype(np.uint8)

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 1
    assert metrics.overlapping_prediction_clusters == 1
    assert metrics.false_negative == 0
    assert metrics.dice_score == 1.0
    assert score_masks(reference, prediction) == metrics


def test_score_masks_accepts_one_voxel_overlap_without_dice_threshold() -> None:
    prediction = np.zeros((20, 20, 20))
    reference = np.zeros_like(prediction)
    prediction[1:15, 1:15, 1:15] = 1
    reference[14:18, 14:18, 14:18] = 1

    metrics = score_masks(prediction, reference)

    assert metrics.detected_reference_clusters == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.dice_score < 0.001


def test_aggregate_metrics_pools_reference_and_prediction_hits_separately() -> None:
    connected = np.zeros((7, 7, 7))
    connected[1:6, 3, 3] = 1
    separated = np.zeros_like(connected)
    separated[1, 3, 3] = 1
    separated[5, 3, 3] = 1
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


def test_evaluate_subjects_writes_overlap_metrics_from_nifti(
    tmp_path: Path, monkeypatch
) -> None:
    prediction = np.zeros((7, 7, 7), dtype=np.uint8)
    prediction[1, 3, 3] = 1
    prediction[5, 3, 3] = 1
    reference = np.zeros_like(prediction)
    reference[1:6, 3, 3] = 255
    affine = np.eye(4)
    affine[0, 3] = 100
    prediction_image = nib.Nifti1Image(prediction, affine)
    reference_image = nib.Nifti1Image(reference, affine)
    prediction_image.header.set_xyzt_units("mm")
    reference_image.header.set_xyzt_units("mm")
    prediction_path = tmp_path / "prediction.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    evaluate.core_io.save_volume(prediction_image, prediction_path)
    evaluate.core_io.save_volume(reference_image, reference_path)
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            subjects=[
                SimpleNamespace(
                    subject_id="subject-1",
                    discriminator_output_path=str(prediction_path),
                )
            ],
        ),
    )
    layout = ExperimentLayout(experiment_dir=tmp_path)

    evaluate.evaluate_subjects(
        EvaluateConfig.model_construct(device="cpu"),
        tmp_path,
        layout,
        [("subject-1", str(reference_path))],
    )

    manifest = EvaluateManifest.read(layout.evaluation_manifest_path())
    assert manifest.subjects[0].metrics.detected_reference_clusters == 1
    assert manifest.subjects[0].metrics.overlapping_prediction_clusters == 2
    assert manifest.aggregate.detected_reference_clusters == 1
    assert manifest.aggregate.overlapping_prediction_clusters == 2
    assert manifest.aggregate.cluster_tpr == 1.0
    assert manifest.aggregate.cluster_precision == 1.0
    assert manifest.aggregate.false_positive == 0
    assert manifest.aggregate.false_negative == 0


def test_evaluate_subjects_requires_reference_mask(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.COMPLETE, subjects=[]),
    )
    config = EvaluateConfig.model_construct(
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        detector_checkpoint_path=tmp_path / "detector.pth",
        student_checkpoint_path=tmp_path / "student.pth",
        device="cpu",
        experiment_dir=None,
    )

    with pytest.raises(ValueError, match="requires a reference mask"):
        evaluate.evaluate_subjects(
            config,
            tmp_path / "dataset",
            ExperimentLayout(experiment_dir=tmp_path / "experiment"),
            [("subject-1", None)],
        )


def test_evaluate_subjects_requires_matching_inference_output(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.COMPLETE, subjects=[]),
    )
    config = EvaluateConfig.model_construct(
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        detector_checkpoint_path=tmp_path / "detector.pth",
        student_checkpoint_path=tmp_path / "student.pth",
        device="cpu",
        experiment_dir=None,
    )

    with pytest.raises(ValueError) as error:
        evaluate.evaluate_subjects(
            config,
            tmp_path / "dataset",
            ExperimentLayout(experiment_dir=tmp_path / "experiment"),
            [("subject-1", "reference")],
        )

    message = str(error.value)
    assert "Inference output is missing" in message
    assert "Rerun the evaluate command" in message


def test_execute_writes_held_out_evaluation_manifest(tmp_path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiment"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    subject = PreprocessedSubject(
        subject_id="subject-1",
        original_volume_path="original-volume",
        brain_mask_path="brain-mask",
        bounding_box=((0, 1), (0, 1), (0, 1)),
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
    for manifest_path in (
        experiment_layout.train_manifest_path(),
        experiment_layout.split_manifest_path(),
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path(),
    ):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.touch()
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1, 1, 1] = 1
    reference[1, 1, 1] = 1

    monkeypatch.setattr(
        configs.TrainManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            split_manifest_fingerprint="fingerprint",
        ),
    )
    monkeypatch.setattr(
        evaluate.SplitManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            preprocessed_manifest_fingerprint="fingerprint",
            test_subject_ids=["subject-1"],
        ),
    )
    monkeypatch.setattr(configs, "content_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(
        evaluate.PreprocessedDatasetManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            subjects=[subject],
        ),
    )
    monkeypatch.setattr(
        evaluate.RawDatasetManifest,
        "read",
        lambda _: SimpleNamespace(
            subjects=[
                SimpleNamespace(
                    subject_id="subject-1", mask_path="raw-reference"
                )
            ]
        ),
    )
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            subjects=[
                SimpleNamespace(
                    subject_id="subject-1",
                    discriminator_output_path="prediction",
                )
            ]
        ),
    )
    inference_amp_settings = []

    def fake_infer_subjects(*args):
        inference_amp_settings.append(args[-1])
        experiment_layout.inference_manifest_path().touch()

    monkeypatch.setattr(evaluate.infer, "infer_subjects", fake_infer_subjects)
    monkeypatch.setattr(
        evaluate.infer,
        "load_inference_models",
        lambda *args: (object(), object()),
    )
    loaded_paths = []

    def load_volume(path):
        loaded_paths.append(path)
        array = prediction if path == "prediction" else reference
        return nib.Nifti1Image(array, np.eye(4))

    monkeypatch.setattr(evaluate.core_io, "load_volume", load_volume)

    evaluate.execute(
        EvaluateConfig.model_validate(
            {
                "experiment_dir": experiment_dir,
                "dataset_dir": dataset_dir,
                "device": "cpu",
                "use_amp": True,
            }
        )
    )

    assert inference_amp_settings == [True]
    manifest = EvaluateManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).evaluation_manifest_path()
    )
    assert [item.subject_id for item in manifest.subjects] == ["subject-1"]
    assert manifest.subjects[0].metrics.detected_reference_clusters == 1
    assert manifest.aggregate.detected_reference_clusters == 1
    assert manifest.aggregate.overlapping_prediction_clusters == 1
    assert manifest.subjects[0].metrics.dice_score == 1.0
    assert manifest.aggregate.dice_score == 1.0
    assert "raw-reference" in loaded_paths
    assert "reference" not in loaded_paths


def test_execute_explicit_checkpoints_evaluates_all_subjects(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    subjects = [
        RawSubject(
            subject_id=f"subject-{index}",
            source_id="source",
            volume_path=f"volume-{index}",
            mask_path=f"reference-{index}",
        )
        for index in range(2)
    ]
    now = timestamp()
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        sources=[
            RawSource(
                input_dir=".",
                volume_pattern="{subject_id}",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=subjects,
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())

    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    detector_checkpoint.touch()
    student_checkpoint.touch()
    output_dir = tmp_path / "evaluation"
    captured = {}

    def fake_infer(config):
        captured["config"] = config

    monkeypatch.setattr(evaluate.infer, "execute", fake_infer)
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(
            status=ManifestStatus.COMPLETE,
            subjects=[
                SimpleNamespace(
                    subject_id=subject.subject_id,
                    discriminator_output_path=f"prediction-{subject.subject_id}",
                )
                for subject in subjects
            ],
        ),
    )
    prediction = np.zeros((3, 3, 3), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1, 1, 1] = 1
    reference[1, 1, 1] = 1
    monkeypatch.setattr(
        evaluate.core_io,
        "load_volume",
        lambda path: nib.Nifti1Image(
            prediction if str(path).startswith("prediction") else reference,
            np.eye(4),
        ),
    )

    evaluate.execute(
        EvaluateConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            detector_checkpoint_path=detector_checkpoint,
            student_checkpoint_path=student_checkpoint,
            use_amp=False,
            device="cpu",
        )
    )

    infer_config = captured["config"]
    assert infer_config.dataset_dir == dataset_dir
    assert infer_config.detector_checkpoint_path == detector_checkpoint
    assert infer_config.student_checkpoint_path == student_checkpoint
    manifest = EvaluateManifest.read(
        ExperimentLayout(experiment_dir=output_dir).evaluation_manifest_path()
    )
    assert [item.subject_id for item in manifest.subjects] == [
        "subject-0",
        "subject-1",
    ]


def test_evaluate_subjects_rejects_incomplete_inference_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    experiment_layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    subject = PreprocessedSubject(
        subject_id="subject-1",
        original_volume_path="original-volume",
        brain_mask_path="brain-mask",
        bounding_box=((0, 1), (0, 1), (0, 1)),
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path="reference", frst_path="frst"
            )
        ],
    )
    monkeypatch.setattr(evaluate.infer, "execute", lambda _: None)
    monkeypatch.setattr(
        evaluate.InferManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.RUNNING, subjects=[]),
    )

    try:
        evaluate.evaluate_subjects(
            cast(EvaluateConfig, SimpleNamespace(device="cpu")),
            dataset_dir,
            experiment_layout,
            [(subject.subject_id, "reference")],
        )
    except ValueError as error:
        message = str(error)
        assert "Cannot evaluate incomplete inference output" in message
        assert "Rerun the evaluate command" in message
    else:
        raise AssertionError("incomplete inference manifest must fail")