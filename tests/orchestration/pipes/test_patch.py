from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from microbleednet.core import utils
from microbleednet.core.common.models import CandidateDetector
from microbleednet.core.datamodels import ExtractedPatches
from microbleednet.orchestration.configs import (
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import (
    PatchManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)
from microbleednet.orchestration.pipes import patch


def _write_subject(
    directory: Path,
    subject_id: str,
    has_microbleed: bool = False,
) -> PreprocessedSubject:
    directory.mkdir(parents=True, exist_ok=True)
    volume_path = directory / f"{subject_id}_volume.nii.gz"
    mask_path = directory / f"{subject_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), volume_path)
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    if has_microbleed:
        mask[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    variants = []
    for variant_index in range(2):
        variant_volume_path = directory / f"{subject_id}_volume_{variant_index}.nii.gz"
        variant_mask_path = directory / f"{subject_id}_mask_{variant_index}.nii.gz"
        variant_frst_path = directory / f"{subject_id}_frst_{variant_index}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), variant_volume_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), variant_mask_path)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), variant_frst_path)
        variants.append(
            PreprocessedVariant(
                volume_path=str(variant_volume_path),
                mask_path=str(variant_mask_path),
                frst_path=str(variant_frst_path),
            )
        )
    return PreprocessedSubject(
        subject_id=subject_id,
        original_volume_path=str(volume_path),
        bounding_box=((0, 1), (0, 1), (0, 1)),
        variants=variants,
    )


def test_execute_materializes_supplied_subjects_and_writes_manifest(
    tmp_path: Path,
) -> None:
    subject = _write_subject(tmp_path / "inputs", "selected", has_microbleed=True)
    _write_subject(tmp_path / "inputs", "not-selected")
    experiment_layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    patch_dir = experiment_layout.patch_dir_path("detector", "train")
    config = NonOverlappingPatchConfig(
        experiment_layout=experiment_layout,
        stage="detector",
        split="train",
        subjects=[subject],
        patch_size=48,
        augmentation_factor=2,
    )

    patch.execute(config)
    records = PatchManifest.read(
        experiment_layout.patch_manifest_path("detector", "train")
    ).records

    assert all(record.patch_index == 0 for record in records)
    assert all(record.has_microbleed for record in records)
    assert np.load(records[0].volume_path).shape == (1, 48, 48, 48)
    assert not (patch_dir / "volumes_not-selected.npy").exists()


def test_execute_uses_configured_augmentation_factor(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path / "inputs", "validation")
    experiment_layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    config = NonOverlappingPatchConfig(
        experiment_layout=experiment_layout,
        stage="teacher",
        split="validation",
        subjects=[subject],
        patch_size=24,
        augmentation_factor=1,
    )

    patch.execute(config)
    records = PatchManifest.read(
        experiment_layout.patch_manifest_path("teacher", "validation")
    ).records

    assert len(records) == 1
    assert np.load(records[0].volume_path).shape == (1, 24, 24, 24)


def test_execute_rejects_unsupported_patch_config(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="supported patch configuration"):
        patch.execute(object())  # pyright: ignore[reportArgumentType]


def test_config_rejects_later_subject_without_mask_before_output(
    tmp_path: Path,
) -> None:
    first = _write_subject(tmp_path / "inputs", "first")
    second = _write_subject(tmp_path / "inputs", "unmasked")
    second = second.model_copy(
        update={
            "variants": [
                second.variants[0].model_copy(update={"mask_path": None})
            ]
        }
    )
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")

    with pytest.raises(ValueError, match="requires a variant mask"):
        NonOverlappingPatchConfig(
            experiment_layout=layout,
            stage="detector",
            split="train",
            subjects=[first, second],
            patch_size=2,
            augmentation_factor=1,
        )

    assert not layout.patch_dir_path("detector", "train").exists()
    assert not layout.patch_manifest_path("detector", "train").exists()


def test_config_rejects_missing_variant_file_before_output(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path / "inputs", "missing")
    Path(subject.variants[0].frst_path).unlink()
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")

    with pytest.raises(ValueError, match="patch input file is missing"):
        NonOverlappingPatchConfig(
            experiment_layout=layout,
            stage="detector",
            split="train",
            subjects=[subject],
            patch_size=2,
            augmentation_factor=1,
        )

    assert not layout.patch_dir_path("detector", "train").exists()
    assert not layout.patch_manifest_path("detector", "train").exists()


def test_config_rejects_unavailable_augmentation_factor(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path / "inputs", "selected")
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")

    with pytest.raises(ValueError, match="augmentation factor exceeds"):
        NonOverlappingPatchConfig(
            experiment_layout=layout,
            stage="detector",
            split="train",
            subjects=[subject],
            patch_size=2,
            augmentation_factor=3,
        )

    assert not layout.patch_dir_path("detector", "train").exists()
    assert not layout.patch_manifest_path("detector", "train").exists()


def test_target_centered_reuses_extractor_for_all_subjects(
    tmp_path: Path, monkeypatch
) -> None:
    subjects = [
        _write_subject(tmp_path / "inputs", "first"),
        _write_subject(tmp_path / "inputs", "second"),
    ]
    detector = CandidateDetector()
    instances = []

    class FakeExtractor:
        def __init__(self, **kwargs):
            instances.append(self)

        def __call__(self, volume, mask, frst):
            return ExtractedPatches(
                volumes=np.expand_dims(volume, axis=0),
                masks=np.expand_dims(mask, axis=0),
                frst=np.expand_dims(frst, axis=0),
            )

    monkeypatch.setattr(patch, "TargetCenteredExtractor", FakeExtractor)
    experiment_layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    config = TargetCenteredPatchConfig(
        experiment_layout=experiment_layout,
        stage="student",
        split="train",
        subjects=subjects,
        patch_size=24,
        augmentation_factor=1,
        probability_threshold=0.5,
        detector=detector,
    )

    assert patch.execute(config) is None
    records = PatchManifest.read(
        experiment_layout.patch_manifest_path("student", "train")
    ).records

    assert len(instances) == 1
    assert len(records) == 2
    manifest = PatchManifest.read(
        experiment_layout.patch_manifest_path("student", "train")
    )
    assert manifest.subject_ids == ["first", "second"]
    assert manifest.probability_threshold == 0.5


def test_target_centered_extractor_uses_supplied_detector(
    tmp_path: Path, monkeypatch
) -> None:
    detector = CandidateDetector()
    extraction_calls = []
    monkeypatch.setattr(
        utils,
        "predict_logits",
        lambda model, volume: torch.zeros((2, 2, 2, 2)),
    )
    monkeypatch.setattr(
        patch.patch_transforms,
        "extract_centered_patches",
        lambda *args: extraction_calls.append(args),
    )
    extractor = patch.TargetCenteredExtractor(
        detector=detector,
        threshold=0.9,
        patch_size=24,
    )
    volume = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    frst = np.ones((2, 2, 2), dtype=np.float64)

    extracted = extractor(volume, mask, frst)

    assert extraction_calls == []
    assert extracted.volumes.shape == (0, 24, 24, 24)
    assert extracted.masks.shape == (0, 24, 24, 24)
    assert extracted.frst.shape == (0, 24, 24, 24)
    assert extracted.volumes.dtype == volume.dtype
    assert extracted.masks.dtype == mask.dtype
    assert extracted.frst.dtype == frst.dtype


def test_target_centered_extracts_candidate(monkeypatch) -> None:
    extractor = patch.TargetCenteredExtractor(
        detector=CandidateDetector(),
        threshold=0.5,
        patch_size=2,
    )
    logits = torch.zeros((2, 2, 2, 2))
    logits[1, 0, 0, 0] = 10
    monkeypatch.setattr(
        utils, "predict_logits", lambda model, volume: logits
    )

    extracted = extractor(
        np.ones((2, 2, 2)), np.zeros((2, 2, 2)), np.ones((2, 2, 2))
    )

    assert extracted.volumes.shape == (1, 2, 2, 2)


def test_execute_skips_subject_with_no_extracted_patches(
    tmp_path: Path, monkeypatch
) -> None:
    subject = _write_subject(tmp_path / "inputs", "empty")
    config = NonOverlappingPatchConfig(
        experiment_layout=ExperimentLayout(experiment_dir=tmp_path),
        stage="detector",
        split="train",
        subjects=[subject],
        patch_size=2,
        augmentation_factor=1,
    )
    empty = np.empty((0, 2, 2, 2))
    monkeypatch.setattr(
        patch,
        "NonOverlappingExtractor",
        lambda size: lambda volume, mask, frst: ExtractedPatches(empty, empty, empty),
    )

    with pytest.raises(ValueError, match="Patch extraction produced no records"):
        patch.execute(config)
    assert not config.experiment_layout.patch_manifest_path(
        "detector", "train"
    ).exists()



