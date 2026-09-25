"""Tests for held-out component scoring."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

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
            subjects=[SimpleNamespace(subject_id="subject-1", output_path="prediction")]
        ),
    )
    monkeypatch.setattr(
        evaluate.infer,
        "infer_subjects",
        lambda *args: experiment_layout.inference_manifest_path().touch(),
    )
    monkeypatch.setattr(
        evaluate.infer,
        "load_inference_models",
        lambda *args: (object(), object()),
    )
    loaded_paths = []

    def load_volume(path):
        loaded_paths.append(path)
        return path

    monkeypatch.setattr(evaluate.core_io, "load_volume", load_volume)
    monkeypatch.setattr(
        evaluate.core_io,
        "nifti_to_numpy",
        lambda volume: prediction if volume == "prediction" else reference,
    )

    evaluate.execute(
        EvaluateConfig.model_validate(
            {
                "experiment_dir": experiment_dir,
                "dataset_dir": dataset_dir,
                "device": "cpu",
            }
        )
    )

    manifest = EvaluateManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).evaluation_manifest_path()
    )
    assert [item.subject_id for item in manifest.subjects] == ["subject-1"]
    assert manifest.subjects[0].metrics.true_positive == 1
    assert manifest.aggregate.true_positive == 1
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
                    output_path=f"prediction-{subject.subject_id}",
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
        lambda path: path,
    )
    monkeypatch.setattr(
        evaluate.core_io,
        "nifti_to_numpy",
        lambda volume: (
            prediction if str(volume).startswith("prediction") else reference
        ),
    )

    evaluate.execute(
        EvaluateConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            detector_checkpoint_path=detector_checkpoint,
            student_checkpoint_path=student_checkpoint,
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