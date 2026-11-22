from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch.nn as nn

from microbleednet.constants import (
    DETECTOR_THRESHOLD,
    DISCRIMINATOR_PATCH_SIZE,
    MAXIMUM_ELLIPTICITY,
    MINIMUM_BRAIN_DISTANCE_MM,
    MINIMUM_VOLUME_MM3,
    PREPROCESSING_AUGMENTATION_FACTOR,
    STUDENT_THRESHOLD,
)
from microbleednet.core.engines import processor
from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import InferConfig
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
    PreprocessedSubject,
    RawDatasetManifest,
)
from microbleednet.orchestration.pipes import infer
from tests.support import preprocessed_subjects, raw_source, raw_subjects
from tests.support_pipes import write_explicit_inputs

IGNORED_LIFECYCLE_FIELDS = {"created_at", "updated_at"}


def postprocess(
    *regions: slice,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    minimum_volume_mm3: float = 2.5,
    maximum_ellipticity: float = 0.2,
    minimum_brain_distance_mm: float = 5.0,
) -> np.ndarray:
    """Post-process a 12^3 candidate mask holding one cube per region slice."""
    candidates = np.zeros((12, 12, 12), dtype=np.uint8)
    for region in regions:
        candidates[region, region, region] = 1
    return processor.postprocess(
        candidates,
        np.ones((12, 12, 12), dtype=np.float32),
        spacing,
        minimum_volume_mm3=minimum_volume_mm3,
        maximum_ellipticity=maximum_ellipticity,
        minimum_brain_distance_mm=minimum_brain_distance_mm,
    )


@pytest.fixture
def cleaned() -> np.ndarray:
    return postprocess(slice(6, 9), slice(0, 1))


def test_postprocess_keeps_interior_component(cleaned: np.ndarray) -> None:
    assert cleaned[7, 7, 7] == 1


def test_postprocess_drops_small_boundary_component(cleaned: np.ndarray) -> None:
    assert cleaned[0, 0, 0] == 0


def test_postprocess_returns_uint8(cleaned: np.ndarray) -> None:
    assert cleaned.dtype == np.uint8


def test_postprocess_uses_physical_spacing_for_component_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[float, float, float] | None] = []
    original_regionprops = processor.regionprops

    def regionprops_with_observed_spacing(labels, **kwargs):
        observed.append(kwargs.get("spacing"))
        return original_regionprops(labels, **kwargs)

    monkeypatch.setattr(processor, "regionprops", regionprops_with_observed_spacing)

    postprocess(slice(5, 8), spacing=(1.0, 1.0, 2.0))

    assert observed == [None, (1.0, 1.0, 2.0)]


def test_postprocess_uses_voxel_centroid_for_distance_index() -> None:
    output = postprocess(
        slice(5, 8),
        spacing=(1.0, 1.0, 2.0),
        minimum_volume_mm3=0.0,
        maximum_ellipticity=1.0,
        minimum_brain_distance_mm=0.0,
    )

    assert output[6, 6, 6] == 1


# execute


@pytest.fixture
def execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake preprocessing, model loading and inference around ``infer.execute``."""
    inputs = write_explicit_inputs(
        tmp_path,
        subjects=[
            *raw_subjects(1),
            raw_subjects(2, source_id="other")[1],
        ],
        sources=[raw_source(), raw_source("other", modality="SWI")],
    )
    state = SimpleNamespace(
        inputs=inputs,
        processed=preprocessed_subjects(2, tmp_path / "processed"),
        models=(cast(Any, object()), cast(Any, object())),
        preprocess_calls=[],
        infer_calls=[],
    )

    def preprocess_subject(*args: Any, **kwargs: Any) -> PreprocessedSubject:
        state.preprocess_calls.append((args, kwargs))
        return state.processed[len(state.preprocess_calls) - 1]

    monkeypatch.setattr(infer, "preprocess_subject", preprocess_subject)
    monkeypatch.setattr(infer, "load_inference_models", lambda *args: state.models)
    monkeypatch.setattr(
        infer, "infer_subjects", lambda *args: state.infer_calls.append(args)
    )
    monkeypatch.setattr(infer, "release_gpu_memory", lambda: None)
    state.config = lambda **overrides: InferConfig(**{**inputs, **overrides})
    return state


@pytest.mark.parametrize("bias_field_correction", ["fast", "n4"])
def test_execute_preprocesses_each_raw_subject_with_its_modality(
    execution: SimpleNamespace, bias_field_correction: str
) -> None:
    config = execution.config(bias_field_correction=bias_field_correction)

    infer.execute(config)

    raw_manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=config.dataset_dir).raw_manifest_path()
    )
    preprocessing_layout = DatasetLayout(dataset_dir=config.output_dir)
    assert execution.preprocess_calls == [
        (
            (
                subject,
                modality,
                preprocessing_layout,
                PREPROCESSING_AUGMENTATION_FACTOR,
            ),
            {"bias_field_correction": bias_field_correction},
        )
        for subject, modality in zip(raw_manifest.subjects, ["QSM", "SWI"])
    ]


def test_execute_infers_preprocessed_subjects_with_loaded_models(
    execution: SimpleNamespace,
) -> None:
    config = execution.config(use_amp=True)

    infer.execute(config)

    assert execution.infer_calls == [
        (
            "cpu",
            ExperimentLayout(experiment_dir=config.output_dir),
            config.detector_checkpoint_path,
            config.student_checkpoint_path,
            execution.processed,
            *execution.models,
            True,
        )
    ]


def test_execute_loads_checkpoints_before_preprocessing(
    execution: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_to_load(*_: Any) -> None:
        raise ApplicationError(category="Checkpoint", summary="invalid checkpoint")

    monkeypatch.setattr(infer, "load_inference_models", fail_to_load)
    config = execution.config()

    with pytest.raises(ApplicationError, match="invalid checkpoint"):
        infer.execute(config)

    assert execution.preprocess_calls == []
    assert (
        not ExperimentLayout(experiment_dir=config.output_dir)
        .inference_manifest_path()
        .exists()
    )


# load_inference_models


class FakeDetector(nn.Module):
    pass


class FakeStudent(nn.Module):
    pass


@pytest.fixture
def model_loading(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(loads=[], releases=0)
    monkeypatch.setattr(infer, "CandidateDetector", FakeDetector)
    monkeypatch.setattr(infer, "CandidateDiscriminatorStudent", FakeStudent)
    monkeypatch.setattr(
        infer,
        "release_gpu_memory",
        lambda: setattr(state, "releases", state.releases + 1),
    )
    return state


def test_load_inference_models_returns_both_loaded_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_loading: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        infer,
        "load_stage_checkpoint",
        lambda model, path, stage: model_loading.loads.append((model, path, stage)),
    )

    detector, student = infer.load_inference_models(
        "cpu", tmp_path / "detector.pth", tmp_path / "student.pth"
    )

    assert isinstance(detector, FakeDetector)
    assert isinstance(student, FakeStudent)
    assert model_loading.loads == [
        (detector, tmp_path / "detector.pth", "detector"),
        (student, tmp_path / "student.pth", "student"),
    ]


def test_load_inference_models_wraps_checkpoint_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_loading: SimpleNamespace
) -> None:
    def fail(*_: Any) -> None:
        raise RuntimeError("invalid checkpoint")

    monkeypatch.setattr(infer.core_io, "load_model_weights", fail)

    with pytest.raises(ApplicationError, match="Could not load the detector"):
        infer.load_inference_models(
            "cpu", tmp_path / "detector.pth", tmp_path / "student.pth"
        )


def test_load_inference_models_preserves_student_checkpoint_error_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_loading: SimpleNamespace
) -> None:
    expected = ApplicationError(
        category="Checkpoint",
        summary="student checkpoint error",
        cause="already structured",
        fix="fix",
    )

    def load_checkpoint(model: nn.Module, path: Path, stage: str) -> None:
        model_loading.loads.append((path, stage))
        if stage == "student":
            raise expected

    monkeypatch.setattr(infer, "load_stage_checkpoint", load_checkpoint)

    with pytest.raises(ApplicationError) as raised:
        infer.load_inference_models(
            "cpu", tmp_path / "detector.pth", tmp_path / "student.pth"
        )

    assert raised.value is expected
    assert model_loading.loads == [
        (tmp_path / "detector.pth", "detector"),
        (tmp_path / "student.pth", "student"),
    ]
    assert model_loading.releases == 1


# infer_subjects


@pytest.fixture
def layout(tmp_path: Path) -> ExperimentLayout:
    return ExperimentLayout(experiment_dir=tmp_path / "output")


@pytest.fixture
def subject_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: ExperimentLayout
) -> SimpleNamespace:
    """Run ``infer_subjects`` over two subjects whose per-subject step is faked."""
    state = SimpleNamespace(
        subjects=preprocessed_subjects(2, tmp_path / "processed"),
        models=(cast(Any, object()), cast(Any, object())),
        calls=[],
        detector_checkpoint=tmp_path / "detector.pth",
        student_checkpoint=tmp_path / "student.pth",
    )

    def infer_subject(*args: Any) -> InferredSubject:
        state.calls.append(args)
        subject_id = args[0].subject_id
        return InferredSubject(
            subject_id=subject_id,
            detector_output_path=f"{subject_id}-detector",
            discriminator_output_path=f"{subject_id}-final",
        )

    monkeypatch.setattr(infer, "infer_subject", infer_subject)
    state.run = lambda: infer.infer_subjects(
        "cpu",
        layout,
        state.detector_checkpoint,
        state.student_checkpoint,
        state.subjects,
        state.models[0],
        state.models[1],
        True,
    )
    return state


def test_infer_subjects_infers_every_subject_with_the_models(
    subject_inference: SimpleNamespace, layout: ExperimentLayout
) -> None:
    subject_inference.run()

    assert subject_inference.calls == [
        (subject, *subject_inference.models, layout, True)
        for subject in subject_inference.subjects
    ]


def test_infer_subjects_writes_manifest_of_results_and_recipe(
    subject_inference: SimpleNamespace, layout: ExperimentLayout
) -> None:
    subject_inference.run()

    expected = InferManifest(
        status=ManifestStatus.COMPLETE,
        device="cpu",
        detector_checkpoint_path=str(subject_inference.detector_checkpoint.resolve()),
        student_checkpoint_path=str(subject_inference.student_checkpoint.resolve()),
        detector_threshold=DETECTOR_THRESHOLD,
        student_threshold=STUDENT_THRESHOLD,
        discriminator_patch_size=DISCRIMINATOR_PATCH_SIZE,
        minimum_volume_mm3=MINIMUM_VOLUME_MM3,
        maximum_ellipticity=MAXIMUM_ELLIPTICITY,
        minimum_brain_distance_mm=MINIMUM_BRAIN_DISTANCE_MM,
        subjects=[
            InferredSubject(
                subject_id=subject.subject_id,
                detector_output_path=f"{subject.subject_id}-detector",
                discriminator_output_path=f"{subject.subject_id}-final",
            )
            for subject in subject_inference.subjects
        ],
    )
    manifest = InferManifest.read(layout.inference_manifest_path())
    assert manifest.model_dump(exclude=IGNORED_LIFECYCLE_FIELDS) == expected.model_dump(
        exclude=IGNORED_LIFECYCLE_FIELDS
    )


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_infer_subjects_wraps_manifest_write_failure(
    subject_inference: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
) -> None:
    def fail(*_: Any) -> None:
        raise failure("read-only")

    monkeypatch.setattr(infer.InferManifest, "write", fail)

    with pytest.raises(
        ApplicationError, match="Could not write the inference manifest"
    ):
        subject_inference.run()


# infer_subject


@pytest.fixture
def subject_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: ExperimentLayout
) -> SimpleNamespace:
    """One real 3^3 subject whose detector, discriminator and postprocess are faked.

    The volume has anisotropic voxels and a 1^3 crop box, so the faked 1^3 stage
    outputs are restored for real into the 3^3 original space.
    """
    (subject,) = preprocessed_subjects(
        1,
        tmp_path / "processed",
        write_files=True,
        shape=(3, 3, 3),
        affine=np.diag([1.0, 2.0, 3.0, 1.0]),
    )
    state = SimpleNamespace(
        subject=subject,
        layout=layout,
        detector_probability=np.full((1, 1, 1), 0.25, dtype=np.float32),
        retained_labels=np.full((1, 1, 1), 3, dtype=np.uint8),
        final_mask=np.full((1, 1, 1), 7, dtype=np.uint8),
        calls={},
        models=(cast(Any, object()), cast(Any, object())),
    )

    def record(name: str, result: np.ndarray):
        def stage(*args: Any) -> np.ndarray:
            state.calls[name] = args
            return result

        return stage

    monkeypatch.setattr(
        infer.core_inference,
        "infer_detector",
        record("detector", state.detector_probability),
    )
    monkeypatch.setattr(
        infer.core_inference,
        "infer_discriminator",
        record("discriminator", state.retained_labels),
    )
    monkeypatch.setattr(
        infer.core_processor, "postprocess", record("postprocess", state.final_mask)
    )
    state.run = lambda use_amp=False: infer.infer_subject(
        subject, state.models[0], state.models[1], layout, use_amp
    )
    return state


def read_output(path: Path | str) -> np.ndarray:
    return nib.load(path).get_fdata()  # pyright: ignore[reportAttributeAccessIssue]


def test_infer_subject_saves_detector_probabilities_in_original_space(
    subject_stages: SimpleNamespace,
) -> None:
    subject_stages.run()

    output = read_output(
        subject_stages.layout.inference_detector_output_path("subject-0")
    )
    expected = np.zeros((3, 3, 3))
    expected[0, 0, 0] = 0.25
    np.testing.assert_array_equal(output, expected)


def test_infer_subject_saves_final_mask_in_original_space(
    subject_stages: SimpleNamespace,
) -> None:
    subject_stages.run()

    output = read_output(
        subject_stages.layout.inference_discriminator_output_path("subject-0")
    )
    expected = np.zeros((3, 3, 3))
    expected[0, 0, 0] = 7
    np.testing.assert_array_equal(output, expected)


def test_infer_subject_returns_resolved_output_paths(
    subject_stages: SimpleNamespace,
) -> None:
    result = subject_stages.run()

    layout = subject_stages.layout
    assert result == InferredSubject(
        subject_id="subject-0",
        detector_output_path=str(
            layout.inference_detector_output_path("subject-0").resolve()
        ),
        discriminator_output_path=str(
            layout.inference_discriminator_output_path("subject-0").resolve()
        ),
    )


@pytest.mark.parametrize("use_amp", [True, False])
def test_infer_subject_runs_models_with_amp_setting(
    subject_stages: SimpleNamespace, use_amp: bool
) -> None:
    subject_stages.run(use_amp)

    detector_args = subject_stages.calls["detector"]
    discriminator_args = subject_stages.calls["discriminator"]
    assert detector_args[0] is subject_stages.models[0]
    assert discriminator_args[0] is subject_stages.models[1]
    assert detector_args[-1] is use_amp
    assert discriminator_args[-1] is use_amp


def test_infer_subject_classifies_detector_candidates(
    subject_stages: SimpleNamespace,
) -> None:
    subject_stages.run()

    (_, _, _, probability, detector_threshold, patch_size, student_threshold, _) = (
        subject_stages.calls["discriminator"]
    )
    assert probability is subject_stages.detector_probability
    assert (detector_threshold, patch_size, student_threshold) == (
        DETECTOR_THRESHOLD,
        DISCRIMINATOR_PATCH_SIZE,
        STUDENT_THRESHOLD,
    )


def test_infer_subject_postprocesses_with_header_voxel_sizes(
    subject_stages: SimpleNamespace,
) -> None:
    subject_stages.run()

    labels, brain_mask, voxel_sizes, *limits = subject_stages.calls["postprocess"]
    assert labels is subject_stages.retained_labels
    np.testing.assert_array_equal(brain_mask, np.ones((3, 3, 3)))
    assert voxel_sizes == (1.0, 2.0, 3.0)
    assert limits == [
        MINIMUM_VOLUME_MM3,
        MAXIMUM_ELLIPTICITY,
        MINIMUM_BRAIN_DISTANCE_MM,
    ]


@pytest.mark.parametrize(
    ("failed_output", "message"),
    [
        ("detector", "Could not write the detector output"),
        ("discriminator", "Could not write the discriminator output"),
    ],
)
def test_infer_subject_wraps_output_write_failure(
    subject_stages: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failed_output: str,
    message: str,
) -> None:
    layout = subject_stages.layout
    failed_path = {
        "detector": layout.inference_detector_output_path("subject-0"),
        "discriminator": layout.inference_discriminator_output_path("subject-0"),
    }[failed_output]
    save_volume = infer.core_io.save_volume

    def save_or_fail(image: nib.Nifti1Image, path: Path) -> None:
        if path == failed_path:
            raise OSError("read-only")
        save_volume(image, path)

    monkeypatch.setattr(infer.core_io, "save_volume", save_or_fail)

    with pytest.raises(ApplicationError, match=message):
        subject_stages.run()
