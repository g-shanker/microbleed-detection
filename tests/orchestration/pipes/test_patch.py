from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from microbleednet.core.common.models import CandidateDetector
from microbleednet.core.datamodels import ExtractedPatches
from microbleednet.orchestration.configs import (
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)
from microbleednet.orchestration.manifests import PreprocessedSubject
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
    return PreprocessedSubject(
        subject_id=subject_id,
        volume_path=str(volume_path),
        mask_path=str(mask_path),
    )


def test_execute_materializes_supplied_subjects_and_returns_records(
    tmp_path: Path,
) -> None:
    subject = _write_subject(tmp_path / "inputs", "selected", has_microbleed=True)
    _write_subject(tmp_path / "inputs", "not-selected")
    patch_dir = tmp_path / "experiment" / "detector" / "train"
    config = NonOverlappingPatchConfig(
        patch_dir=patch_dir,
        subjects=[subject],
        patch_size=48,
        augmentation_factor=2,
    )

    records = patch.execute(
        config,
    )

    assert all(record.patch_index == 0 for record in records)
    assert all(record.has_microbleed for record in records)
    assert not records[0].augmented
    assert records[1].augmented
    assert np.load(records[0].volume_path).shape == (1, 48, 48, 48)
    assert not (patch_dir / "volumes_not-selected.npy").exists()


def test_execute_uses_configured_augmentation_factor(tmp_path: Path) -> None:
    subject = _write_subject(tmp_path / "inputs", "validation")
    patch_dir = tmp_path / "experiment" / "teacher" / "validation"
    config = NonOverlappingPatchConfig(
        patch_dir=patch_dir,
        subjects=[subject],
        patch_size=24,
        augmentation_factor=1,
    )

    records = patch.execute(
        config,
    )

    assert len(records) == 1
    assert not records[0].augmented
    assert np.load(records[0].volume_path).shape == (1, 24, 24, 24)


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

        def __call__(self, volume, mask):
            return ExtractedPatches(
                volumes=np.expand_dims(volume, axis=0),
                masks=np.expand_dims(mask, axis=0),
            )

    monkeypatch.setattr(patch, "TargetCenteredExtractor", FakeExtractor)
    patch_dir = tmp_path / "experiment" / "student" / "train"
    config = TargetCenteredPatchConfig(
        patch_dir=patch_dir,
        subjects=subjects,
        patch_size=24,
        augmentation_factor=1,
        probability_threshold=0.5,
        detector=detector,
    )

    records = patch.execute(config)

    assert len(instances) == 1
    assert len(records) == 2


def test_target_centered_extractor_uses_supplied_detector(
    tmp_path: Path, monkeypatch
) -> None:
    detector = CandidateDetector()
    monkeypatch.setattr(
        patch.core_processor,
        "infer",
        lambda model, volume: torch.zeros((2, 2, 2, 2)),
    )
    extractor = patch.TargetCenteredExtractor(
        detector=detector,
        threshold=0.9,
        patch_size=24,
    )
    volume = np.ones((2, 2, 2))
    mask = np.zeros((2, 2, 2), dtype=np.uint8)

    extracted = extractor(volume, mask)
    assert extracted.volumes.size == 0
    assert extracted.masks.size == 0
    extracted = extractor(volume, mask)
    assert extracted.volumes.size == 0
    assert extracted.masks.size == 0


def test_target_centered_extracts_candidate(monkeypatch) -> None:
    extractor = patch.TargetCenteredExtractor(
        detector=CandidateDetector(),
        threshold=0.5,
        patch_size=2,
    )
    logits = torch.zeros((2, 2, 2, 2))
    logits[1, 0, 0, 0] = 10
    monkeypatch.setattr(patch.core_processor, "infer", lambda model, volume: logits)

    extracted = extractor(np.ones((2, 2, 2)), np.zeros((2, 2, 2)))

    assert extracted.volumes.shape == (1, 2, 2, 2)


def test_execute_skips_subject_with_no_extracted_patches(
    tmp_path: Path, monkeypatch
) -> None:
    subject = _write_subject(tmp_path / "inputs", "empty")
    config = NonOverlappingPatchConfig(
        patch_dir=tmp_path / "patches",
        subjects=[subject],
        patch_size=2,
        augmentation_factor=1,
    )
    empty = np.empty((0, 2, 2, 2))
    monkeypatch.setattr(
        patch,
        "NonOverlappingExtractor",
        lambda size: lambda volume, mask: ExtractedPatches(empty, empty),
    )

    assert patch.execute(config) == []

