from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

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


def _write_raw_dataset(
    dataset_dir: Path, with_mask: bool = True, count: int = 1
) -> None:
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    fixture_shape = (32, 32, 32)
    subjects = []
    for index in range(count):
        subject_id = "source_masked" if count == 1 else f"source_masked_{index}"
        volume_path = raw_dir / f"{subject_id}_volume.nii.gz"
        mask_path = raw_dir / f"{subject_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(fixture_shape), np.eye(4)), volume_path)
        if with_mask:
            nib.save(nib.Nifti1Image(np.ones(fixture_shape), np.eye(4)), mask_path)
        subjects.append(
            RawSubject(
                subject_id=subject_id,
                source_id="source",
                volume_path=str(volume_path),
                mask_path=str(mask_path) if with_mask else None,
            )
        )
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

    def fake_preprocess(preprocess_input):
        assert preprocess_input.modality == "QSM"
        fixture_shape = (32, 32, 32)
        output_mask = np.ones(fixture_shape, dtype=np.uint8)
        return SimpleNamespace(
            volume=np.ones(fixture_shape),
            mask=output_mask,
            affine=np.eye(4),
            bounding_box=((0, 32), (0, 32), (0, 32)),
        )

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
    assert variant.volume_path is not None
    assert Path(variant.volume_path).is_file()
    assert variant.mask_path is not None
    assert Path(variant.mask_path).is_file()
    assert variant.frst_path is not None
    assert Path(variant.frst_path).is_file()


def test_execute_writes_maskless_subject_without_mask_output(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir, with_mask=False)

    def fake_preprocess(preprocess_input):
        assert preprocess_input.mask is None
        assert preprocess_input.modality == "QSM"
        fixture_shape = (32, 32, 32)
        return SimpleNamespace(
            volume=np.ones(fixture_shape),
            mask=None,
            affine=np.eye(4),
            bounding_box=((0, 32), (0, 32), (0, 32)),
        )

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


def test_execute_resume_skips_completed_subjects_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir, count=2)
    config = PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
    calls = 0

    def fail_on_second_subject(preprocess_input):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return SimpleNamespace(
            volume=np.ones((32, 32, 32)),
            mask=np.ones((32, 32, 32), dtype=np.uint8),
            affine=np.eye(4),
            bounding_box=((0, 32), (0, 32), (0, 32)),
        )

    monkeypatch.setattr(
        preprocess.processor, "preprocess", fail_on_second_subject
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        preprocess.execute(config)

    layout = DatasetLayout(dataset_dir=dataset_dir)
    checkpoint = PreprocessedDatasetManifest.read(
        layout.preprocessed_manifest_path()
    )
    assert checkpoint.status is ManifestStatus.RUNNING
    assert [subject.subject_id for subject in checkpoint.subjects] == [
        "source_masked_0"
    ]

    calls = 0
    preprocess.execute(config.model_copy(update={"resume": True}))

    completed = PreprocessedDatasetManifest.read(
        layout.preprocessed_manifest_path()
    )
    assert [subject.subject_id for subject in completed.subjects] == [
        "source_masked_0",
        "source_masked_1",
    ]
    assert calls == 1


def test_execute_resume_accepts_changed_augmentation_factor(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir)
    config = PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)

    monkeypatch.setattr(
        preprocess.processor,
        "preprocess",
        lambda preprocess_input: SimpleNamespace(
            volume=np.ones((32, 32, 32)),
            mask=np.ones((32, 32, 32), dtype=np.uint8),
            affine=np.eye(4),
            bounding_box=((0, 32), (0, 32), (0, 32)),
        ),
    )
    preprocess.execute(config)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest.read(manifest_path).model_copy(
        update={"status": ManifestStatus.RUNNING}
    ).write(manifest_path)

    with pytest.raises(ValueError, match="incomplete variants"):
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=2, resume=True)


def test_execute_resume_rejects_changed_raw_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir)
    config = PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
    monkeypatch.setattr(
        preprocess.processor,
        "preprocess",
        lambda preprocess_input: SimpleNamespace(
            volume=np.ones((32, 32, 32)),
            mask=np.ones((32, 32, 32), dtype=np.uint8),
            affine=np.eye(4),
            bounding_box=((0, 32), (0, 32), (0, 32)),
        ),
    )
    preprocess.execute(config)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    PreprocessedDatasetManifest.read(manifest_path).model_copy(
        update={"status": ManifestStatus.RUNNING}
    ).write(manifest_path)

    raw_manifest_path = DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    raw_manifest = RawDatasetManifest.read(raw_manifest_path)
    raw_manifest = raw_manifest.model_copy(
        update={
            "subjects": [
                *raw_manifest.subjects,
                raw_manifest.subjects[0].model_copy(
                    update={"subject_id": "source_masked_new"}
                ),
            ]
        }
    )
    raw_manifest.write(raw_manifest_path)

    with pytest.raises(ValueError, match="Manifest fingerprint does not match"):
        PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1, resume=True)
