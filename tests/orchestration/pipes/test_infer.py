from pathlib import Path

import numpy as np

from microbleednet.core.engines import processor
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import (
    InferManifest,
    ManifestStatus,
)


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
