from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch.nn as nn

from microbleednet.constants import DETECTOR_STAGE, STUDENT_STAGE
from microbleednet.core.engines import processor
from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import (
    InferConfig,
)
from microbleednet.orchestration.layouts import (
    DatasetLayout,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    InferManifest,
    ManifestStatus,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    timestamp,
)
from microbleednet.orchestration.pipes import infer


def test_cleanup_candidates_rejects_small_and_boundary_components() -> None:
    volume = np.ones((12, 12, 12), dtype=np.float32)
    candidate_mask = np.zeros_like(volume, dtype=np.uint8)
    candidate_mask[6:9, 6:9, 6:9] = 1
    candidate_mask[0, 0, 0] = 1

    output = processor.postprocess(
        candidate_mask,
        volume,
        (1.0, 1.0, 1.0),
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
    )

    assert output[7, 7, 7] == 1
    assert output[0, 0, 0] == 0
    assert output.dtype == np.uint8


def test_postprocess_uses_physical_spacing_for_component_shape(
    monkeypatch,
) -> None:
    observed: list[tuple[float, float, float] | None] = []
    original_regionprops = processor.regionprops

    def regionprops_with_observed_spacing(labels, **kwargs):
        observed.append(kwargs.get("spacing"))
        return original_regionprops(labels, **kwargs)

    monkeypatch.setattr(processor, "regionprops", regionprops_with_observed_spacing)

    volume = np.ones((12, 12, 12), dtype=np.float32)
    candidate_mask = np.zeros_like(volume, dtype=np.uint8)
    candidate_mask[5:8, 5:8, 5:8] = 1

    processor.postprocess(
        candidate_mask,
        volume,
        (1.0, 1.0, 2.0),
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
    )

    assert observed == [None, (1.0, 1.0, 2.0)]


def test_postprocess_uses_voxel_centroid_for_distance_index() -> None:
    volume = np.ones((12, 12, 12), dtype=np.float32)
    candidate_mask = np.zeros_like(volume, dtype=np.uint8)
    candidate_mask[5:8, 5:8, 5:8] = 1

    output = processor.postprocess(
        candidate_mask,
        volume,
        (1.0, 1.0, 2.0),
        minimum_volume_mm3=0.0,
        maximum_ellipticity=1.0,
        minimum_brain_distance_mm=0.0,
    )

    assert output[6, 6, 6] == 1


def test_inference_manifest_round_trip(tmp_path: Path) -> None:
    layout = ExperimentLayout(experiment_dir=tmp_path)
    manifest = InferManifest(
        status=ManifestStatus.COMPLETE,
        created_at="now",
        updated_at="now",
        device="cpu",
        detector_checkpoint_path="detector.pth",
        student_checkpoint_path="student.pth",
        detector_threshold=0.5,
        student_threshold=0.5,
        discriminator_patch_size=24,
        minimum_volume_mm3=2.5,
        maximum_ellipticity=0.2,
        minimum_brain_distance_mm=5.0,
        subjects=[],
    )
    manifest.write(layout.inference_manifest_path())

    loaded = InferManifest.read(layout.inference_manifest_path())
    assert loaded.manifest_type == "inference"
    assert loaded.status is ManifestStatus.COMPLETE


def test_execute_loads_checkpoints_before_preprocessing(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    detector_checkpoint.touch()
    student_checkpoint.touch()
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[
            RawSource(
                input_dir=str(tmp_path),
                volume_pattern="{subject_id}",
                source_id="source",
                modality="QSM",
            )
        ],
        subjects=[
            RawSubject(
                subject_id="subject-1",
                source_id="source",
                volume_path="volume",
            )
        ],
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    preprocessing_calls = []
    monkeypatch.setattr(
        infer,
        "load_inference_models",
        lambda *_: (_ for _ in ()).throw(
            ApplicationError(
                category="Checkpoint",
                summary="invalid checkpoint",
            )
        ),
    )
    monkeypatch.setattr(
        infer,
        "preprocess_subject",
        lambda *args: preprocessing_calls.append(args),
    )
    config = InferConfig(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        detector_checkpoint_path=detector_checkpoint,
        student_checkpoint_path=student_checkpoint,
        device="cpu",
    )

    with pytest.raises(ApplicationError, match="invalid checkpoint"):
        infer.execute(config)

    assert preprocessing_calls == []
    assert not ExperimentLayout(
        experiment_dir=output_dir
    ).inference_manifest_path().exists()


def test_execute_writes_inference_manifest(tmp_path: Path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiment"
    layout = ExperimentLayout(experiment_dir=experiment_dir)
    for stage in (DETECTOR_STAGE, STUDENT_STAGE):
        checkpoint = layout.best_checkpoint_path(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")

    subject = PreprocessedSubject(
        subject_id="subject-1",
        original_volume_path="original-volume",
        bounding_box=((0, 3), (0, 3), (0, 3)),
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path="mask", frst_path="frst"
            )
        ],
    )
    volume_image = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    saved = {}
    restore_calls = []

    class FakeModel(nn.Module):
        def forward(self, inputs):
            return inputs

    monkeypatch.setattr(infer.core_io, "load_volume", lambda _: volume_image)
    monkeypatch.setattr(
        infer.core_io,
        "nifti_to_numpy",
        lambda _: np.ones((3, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        infer.core_inference,
        "infer_detector",
        lambda *args: np.ones((3, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        infer.core_inference,
        "infer_discriminator",
        lambda *args: np.ones((3, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        infer.core_processor,
        "postprocess",
        lambda *args: np.ones((3, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(infer.core_io, "numpy_to_nifti", lambda array, _: array)
    monkeypatch.setattr(
        infer.core_processor.volume_ops,
        "restore_cropped_volume",
        lambda prediction, bounding_box, original_volume: restore_calls.append(
            (prediction, bounding_box, original_volume)
        )
        or volume_image,
    )
    monkeypatch.setattr(
        infer.core_io,
        "save_volume",
        lambda image, path: saved.update({str(path): image}),
    )
    monkeypatch.setattr(infer, "release_gpu_memory", lambda: None)

    infer.infer_subjects(
        "cpu",
        layout,
        layout.best_checkpoint_path(DETECTOR_STAGE),
        layout.best_checkpoint_path(STUDENT_STAGE),
        [subject],
        cast(Any, FakeModel()),
        cast(Any, FakeModel()),
    )

    manifest = InferManifest.read(layout.inference_manifest_path())
    assert [item.subject_id for item in manifest.subjects] == ["subject-1"]
    assert len(saved) == 1
    assert restore_calls[0][1] == ((0, 3), (0, 3), (0, 3))


def test_load_inference_models_wraps_checkpoint_load_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel(nn.Module):
        def forward(self, inputs):
            return inputs

    monkeypatch.setattr(infer, "CandidateDetector", FakeModel)
    monkeypatch.setattr(infer, "CandidateDiscriminatorStudent", FakeModel)
    monkeypatch.setattr(
        infer.core_io,
        "load_model_weights",
        lambda *args: (_ for _ in ()).throw(RuntimeError("invalid checkpoint")),
    )
    monkeypatch.setattr(infer, "release_gpu_memory", lambda: None)

    with pytest.raises(ValueError, match="Could not load the detector"):
        infer.load_inference_models(
            "cpu",
            tmp_path / "detector.pth",
            tmp_path / "student.pth",
        )


def test_load_inference_models_returns_both_loaded_models(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeDetector(nn.Module):
        pass

    class FakeStudent(nn.Module):
        pass

    loads = []
    monkeypatch.setattr(infer, "CandidateDetector", FakeDetector)
    monkeypatch.setattr(infer, "CandidateDiscriminatorStudent", FakeStudent)
    monkeypatch.setattr(
        infer,
        "load_stage_checkpoint",
        lambda model, path, stage: loads.append((model, path, stage)),
    )

    detector, student = infer.load_inference_models(
        "cpu",
        tmp_path / "detector.pth",
        tmp_path / "student.pth",
    )

    assert isinstance(detector, FakeDetector)
    assert isinstance(student, FakeStudent)
    assert loads == [
        (detector, tmp_path / "detector.pth", "detector"),
        (student, tmp_path / "student.pth", "student"),
    ]


def test_load_inference_models_preserves_student_checkpoint_error(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel(nn.Module):
        def forward(self, inputs):
            return inputs

    expected = ApplicationError(
        category="Checkpoint",
        summary="student checkpoint error",
        cause="already structured",
        fix="fix",
    )
    loads = []
    releases = []

    def load_checkpoint(model, path, stage) -> None:
        loads.append((path, stage))
        if stage == "student":
            raise expected

    monkeypatch.setattr(infer, "CandidateDetector", FakeModel)
    monkeypatch.setattr(infer, "CandidateDiscriminatorStudent", FakeModel)
    monkeypatch.setattr(
        infer,
        "load_stage_checkpoint",
        load_checkpoint,
    )
    monkeypatch.setattr(
        infer, "release_gpu_memory", lambda: releases.append("release")
    )

    with pytest.raises(ApplicationError) as raised:
        infer.load_inference_models(
            "cpu",
            tmp_path / "detector.pth",
            tmp_path / "student.pth",
        )

    assert raised.value is expected
    assert loads == [
        (tmp_path / "detector.pth", "detector"),
        (tmp_path / "student.pth", "student"),
    ]
    assert releases == ["release"]


def test_infer_subjects_wraps_manifest_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel(nn.Module):
        def forward(self, inputs):
            return inputs

    monkeypatch.setattr(
        infer.InferManifest,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("read-only")),
    )
    monkeypatch.setattr(infer, "release_gpu_memory", lambda: None)

    with pytest.raises(ValueError, match="Could not write the inference manifest"):
        infer.infer_subjects(
            "cpu",
            ExperimentLayout(experiment_dir=tmp_path),
            tmp_path / "detector.pth",
            tmp_path / "student.pth",
            [],
            cast(Any, FakeModel()),
            cast(Any, FakeModel()),
        )


def test_infer_subject_wraps_output_write_failure(tmp_path: Path, monkeypatch) -> None:
    subject = PreprocessedSubject(
        subject_id="subject-1",
        original_volume_path="original-volume",
        bounding_box=((0, 3), (0, 3), (0, 3)),
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path=None, frst_path="frst"
            )
        ],
    )
    volume_image = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    monkeypatch.setattr(infer.core_io, "load_volume", lambda _: volume_image)
    monkeypatch.setattr(
        infer.core_io,
        "nifti_to_numpy",
        lambda _: np.ones((3, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        infer.core_inference,
        "infer_detector",
        lambda *args: np.ones((3, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        infer.core_inference,
        "infer_discriminator",
        lambda *args: np.ones((3, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        infer.core_processor,
        "postprocess",
        lambda *args: np.ones((3, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        infer.core_processor.volume_ops,
        "restore_cropped_volume",
        lambda *args: volume_image,
    )
    monkeypatch.setattr(
        infer.core_io,
        "save_volume",
        lambda *_args: (_ for _ in ()).throw(OSError("read-only")),
    )

    with pytest.raises(ValueError, match="Could not write the inference output"):
        infer.infer_subject(
            subject,
            cast(Any, object()),
            cast(Any, object()),
            ExperimentLayout(experiment_dir=tmp_path),
        )


def test_execute_preprocesses_raw_subjects(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    detector_checkpoint = tmp_path / "detector.pth"
    student_checkpoint = tmp_path / "student.pth"
    calls = []
    dataset_dir.mkdir()
    detector_checkpoint.touch()
    student_checkpoint.touch()
    raw_subject = RawSubject(
        subject_id="subject-1",
        source_id="source",
        volume_path="volume",
    )
    preprocessed_subject = PreprocessedSubject(
        subject_id="subject-1",
        original_volume_path="original-volume",
        bounding_box=((0, 1), (0, 1), (0, 1)),
        variants=[
            PreprocessedVariant(
                volume_path="processed-volume",
                mask_path=None,
                frst_path="processed-frst",
            )
        ],
    )
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
        subjects=[raw_subject],
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())

    monkeypatch.setattr(infer, "preprocess_subject", lambda *args: preprocessed_subject)
    detector = cast(Any, object())
    student = cast(Any, object())
    monkeypatch.setattr(
        infer, "load_inference_models", lambda *args: (detector, student)
    )
    monkeypatch.setattr(
        infer, "infer_subjects", lambda *args: calls.append(args)
    )

    config = InferConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        detector_checkpoint_path=detector_checkpoint,
        student_checkpoint_path=student_checkpoint,
        device="cpu",
    )

    infer.execute(config)

    assert calls == [
        (
            "cpu",
            ExperimentLayout(experiment_dir=output_dir),
            detector_checkpoint,
            student_checkpoint,
            [preprocessed_subject],
            detector,
            student,
        )
    ]


