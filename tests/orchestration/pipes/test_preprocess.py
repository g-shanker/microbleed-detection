from pathlib import Path

import nibabel as nib
import numpy as np

from microbleednet.orchestration.configs import PreprocessConfig
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    timestamp,
)
from microbleednet.orchestration.pipes import preprocess


def test_layout_uses_frst_suffix_for_frst_variants(tmp_path: Path) -> None:
    layout = DatasetLayout(
        dataset_dir=tmp_path,
        volume_suffix=".volume",
        mask_suffix=".mask",
        frst_suffix=".frst",
    )

    assert layout.variant_volume_path("subject", 0).name == "subject_variant_0.volume"
    assert layout.variant_mask_path("subject", 0).name == "subject_variant_0.mask"
    assert layout.variant_frst_path("subject", 0).name == "subject_variant_0.frst"


def _write_raw_dataset(dataset_dir: Path, with_mask: bool = True) -> None:
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    volume_path = raw_dir / "masked_volume.nii.gz"
    mask_path = raw_dir / "masked_mask.nii.gz"
    fixture_shape = (32, 32, 32)
    nib.save(nib.Nifti1Image(np.ones(fixture_shape), np.eye(4)), volume_path)
    if with_mask:
        nib.save(nib.Nifti1Image(np.ones(fixture_shape), np.eye(4)), mask_path)
    subjects = [
        RawSubject(
            subject_id="source_masked",
            source_id="source",
            volume_path=str(volume_path),
            mask_path=str(mask_path) if with_mask else None,
        )
    ]
    now = timestamp()
    manifest = RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        sources=[
            RawSource(
                input_dir=str(raw_dir),
                mask_dir=str(raw_dir) if with_mask else None,
                volume_pattern="{subject_id}_volume.nii.gz",
                mask_pattern="{subject_id}_mask.nii.gz" if with_mask else None,
                source_id="source",
                modality="QSM",
                added_on=now,
            )
        ],
        subjects=subjects,
    )
    layout = DatasetLayout(dataset_dir=dataset_dir)
    manifest.write(layout.raw_manifest_path())


def test_execute_writes_volumes_masks_and_complete_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir)

    def fake_preprocess(preprocess_input) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert preprocess_input.modality == "QSM"
        fixture_shape = (32, 32, 32)
        output_mask = np.ones(fixture_shape, dtype=np.uint8)
        return np.ones(fixture_shape), output_mask, np.eye(4)

    monkeypatch.setattr(preprocess.processor, "preprocess", fake_preprocess)

    preprocess.execute(
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=2)
    )

    layout = DatasetLayout(dataset_dir=dataset_dir)
    manifest = PreprocessedDatasetManifest.read(layout.preprocessed_manifest_path())
    assert manifest.status is ManifestStatus.COMPLETE
    assert [subject.subject_id for subject in manifest.subjects] == ["source_masked"]
    subject = manifest.subjects[0]
    assert manifest.augmentation_factor == 2
    assert len(subject.variants) == 2
    variant = subject.variants[0]
    assert Path(variant.volume_path).is_file()
    assert Path(variant.mask_path).is_file()
    assert variant.frst_path is not None
    assert Path(variant.frst_path).is_file()


def test_execute_tracks_subject_progress(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir)
    tracked = {}

    def fake_track(items, description):
        tracked["items"] = items
        tracked["description"] = description
        return items

    monkeypatch.setattr(preprocess.progress, "track", fake_track)
    monkeypatch.setattr(
        preprocess.processor,
        "preprocess",
        lambda preprocess_input: (np.ones((32, 32, 32)), None, np.eye(4)),
    )

    preprocess.execute(
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
    )

    assert tracked["description"] == "Preprocessing subjects"
    assert len(tracked["items"]) == 1


def test_execute_writes_maskless_subject_without_mask_output(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir, with_mask=False)

    def fake_preprocess(preprocess_input) -> tuple[np.ndarray, None, np.ndarray]:
        assert preprocess_input.mask is None
        assert preprocess_input.modality == "QSM"
        fixture_shape = (32, 32, 32)
        return np.ones(fixture_shape), None, np.eye(4)

    monkeypatch.setattr(preprocess.processor, "preprocess", fake_preprocess)

    preprocess.execute(
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
    )

    layout = DatasetLayout(dataset_dir=dataset_dir)
    manifest = PreprocessedDatasetManifest.read(layout.preprocessed_manifest_path())
    variant = manifest.subjects[0].variants[0]
    assert variant.mask_path is None
    assert Path(variant.volume_path).is_file()
    assert variant.frst_path is not None
    assert Path(variant.frst_path).is_file()
    assert not layout.variant_mask_path("source_masked", 0).exists()
