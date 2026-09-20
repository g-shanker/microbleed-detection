from pathlib import Path

import nibabel as nib
import numpy as np
import torch.nn as nn

from microbleednet.core.engines import processor
from microbleednet.orchestration.configs import InferConfig, InferExperimentConfig
from microbleednet.orchestration.layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    InferManifest,
    ManifestStatus,
    PreprocessedSubject,
    PreprocessedVariant,
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


def test_execute_writes_inference_manifest(tmp_path: Path, monkeypatch) -> None:
    experiment_dir = tmp_path / "experiment"
    layout = ExperimentLayout(experiment_dir=experiment_dir)
    for stage in (DETECTOR_STAGE, STUDENT_STAGE):
        checkpoint = layout.best_checkpoint_path(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")

    subject = PreprocessedSubject(
        subject_id="subject-1",
        variants=[
            PreprocessedVariant(
                volume_path="volume", mask_path="mask", frst_path="frst"
            )
        ],
    )
    volume_image = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    saved = {}

    class FakeModel(nn.Module):
        def forward(self, inputs):
            return inputs

    monkeypatch.setattr(infer, "CandidateDetector", FakeModel)
    monkeypatch.setattr(infer, "CandidateDiscriminatorStudent", FakeModel)
    monkeypatch.setattr(infer.core_io, "load_model_weights", lambda *args: None)
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
        infer.core_io,
        "save_volume",
        lambda image, path: saved.update({str(path): image}),
    )
    monkeypatch.setattr(infer, "release_gpu_memory", lambda: None)

    infer.execute(
        InferConfig(
            experiment=InferExperimentConfig(
                experiment_dir=experiment_dir,
                subjects=[subject],
            ),
            device="cpu",
        )
    )

    manifest = InferManifest.read(layout.inference_manifest_path())
    assert [item.subject_id for item in manifest.subjects] == ["subject-1"]
    assert len(saved) == 1
