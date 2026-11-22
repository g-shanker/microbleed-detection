from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import PreprocessConfig
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    content_fingerprint,
)
from microbleednet.orchestration.pipes import preprocess
from tests.support import (
    preprocessed_subjects,
    raw_source,
    raw_subjects,
    write_preprocessed_manifest,
    write_raw_manifest,
)

SHAPE = (16, 16, 16)


def _output(with_mask: bool = True) -> SimpleNamespace:
    """Stand in for the processor result without running FSL."""
    return SimpleNamespace(
        volume=np.ones(SHAPE),
        mask=np.ones(SHAPE, dtype=np.uint8) if with_mask else None,
        brain_mask=np.ones(SHAPE, dtype=np.uint8),
        affine=np.eye(4),
        bounding_box=tuple((0, size) for size in SHAPE),
    )


def _write_raw_dataset(root: Path, count: int = 1, with_mask: bool = True) -> Path:
    dataset_dir = root / "dataset"
    subjects = raw_subjects(
        count,
        source_id="source",
        directory=dataset_dir / "raw",
        with_mask=with_mask,
        shape=SHAPE,
        dtype=np.float32,
    )
    write_raw_manifest(dataset_dir, subjects, sources=[raw_source("source")])
    return dataset_dir


def _read_manifest(dataset_dir: Path) -> PreprocessedDatasetManifest:
    return PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    )


@pytest.fixture(scope="module")
def executed(tmp_path_factory: pytest.TempPathFactory):
    """Run one two-variant preprocessing pass and record processor inputs."""
    dataset_dir = _write_raw_dataset(tmp_path_factory.mktemp("preprocess"))
    inputs = []

    def fake_preprocess(preprocess_input):
        inputs.append(preprocess_input)
        return _output()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preprocess.processor, "preprocess", fake_preprocess)
        preprocess.execute(
            PreprocessConfig(
                dataset_dir=dataset_dir,
                augmentation_factor=2,
                bias_field_correction="n4",
                seed=42,
            )
        )
    return _read_manifest(dataset_dir), inputs


def test_layout_uses_frst_suffix_for_frst_variants(tmp_path: Path) -> None:
    layout = DatasetLayout(
        dataset_dir=tmp_path,
        volume_suffix=".volume",
        mask_suffix=".mask",
        frst_suffix=".frst",
    )

    assert layout.variant_volume_path("subject", 0).name == "subject_variant_0.volume"
    assert layout.variant_mask_path("subject", 0).name == "subject_variant_0.mask"
    assert layout.brain_mask_path("subject").name == "subject_brain_mask.mask"
    assert layout.variant_frst_path("subject", 0).name == "subject_variant_0.frst"


def test_execute_writes_complete_manifest(executed) -> None:
    manifest, _ = executed

    assert manifest.status is ManifestStatus.COMPLETE
    assert [subject.subject_id for subject in manifest.subjects] == ["subject-0"]
    assert manifest.augmentation_factor == 2
    assert manifest.seed == 42
    assert manifest.bias_field_correction == "n4"
    assert len(manifest.subjects[0].variants) == 2


def test_execute_passes_modality_mask_and_bias_correction_to_processor(
    executed,
) -> None:
    _, inputs = executed

    assert len(inputs) == 1
    assert inputs[0].modality == "QSM"
    assert inputs[0].bias_field_correction == "n4"
    assert inputs[0].mask is not None


def test_execute_writes_brain_mask(executed) -> None:
    manifest, _ = executed

    assert Path(manifest.subjects[0].brain_mask_path).is_file()


@pytest.mark.parametrize(
    ("attribute", "dtype"),
    [("volume_path", np.float32), ("frst_path", np.float32), ("mask_path", np.uint8)],
)
def test_execute_writes_typed_variant_outputs(
    executed, attribute: str, dtype: type
) -> None:
    manifest, _ = executed

    for variant in manifest.subjects[0].variants:
        path = getattr(variant, attribute)
        assert path is not None
        assert preprocess.io.load_volume(path).get_data_dtype() == np.dtype(dtype)


def test_execute_adds_subject_context_to_preprocessing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    dataset_dir = _write_raw_dataset(tmp_path)
    failure = ApplicationError(
        category="Input data", summary="Volume and mask affines do not match"
    )

    def fail(_):
        raise failure

    monkeypatch.setattr(preprocess.processor, "preprocess", fail)

    with pytest.raises(ApplicationError) as raised:
        preprocess.execute(
            PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
        )

    assert raised.value is failure
    assert failure.context == {"subject_id": "subject-0", "stage": "preprocessing"}
    assert "subject_id=subject-0" in caplog.text


def test_execute_writes_maskless_subject_without_mask_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _write_raw_dataset(tmp_path, with_mask=False)
    inputs = []

    def fake_preprocess(preprocess_input):
        inputs.append(preprocess_input)
        return _output(with_mask=False)

    monkeypatch.setattr(preprocess.processor, "preprocess", fake_preprocess)

    preprocess.execute(PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1))

    variant = _read_manifest(dataset_dir).subjects[0].variants[0]
    assert inputs[0].mask is None
    assert variant.mask_path is None
    assert Path(variant.volume_path).is_file()
    assert Path(variant.frst_path).is_file()
    assert (
        not DatasetLayout(dataset_dir=dataset_dir)
        .variant_mask_path("subject-0", 0)
        .exists()
    )


def test_execute_resume_skips_completed_subjects_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _write_raw_dataset(tmp_path, count=2)
    config = PreprocessConfig(dataset_dir=dataset_dir, augmentation_factor=1)
    calls = 0

    def fail_on_second_subject(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return _output()

    monkeypatch.setattr(preprocess.processor, "preprocess", fail_on_second_subject)

    with pytest.raises(RuntimeError, match="interrupted"):
        preprocess.execute(config)

    checkpoint = _read_manifest(dataset_dir)
    assert checkpoint.status is ManifestStatus.RUNNING
    assert [subject.subject_id for subject in checkpoint.subjects] == ["subject-0"]

    calls = 0
    preprocess.execute(config.model_copy(update={"resume": True}))

    assert [subject.subject_id for subject in _read_manifest(dataset_dir).subjects] == [
        "subject-0",
        "subject-1",
    ]
    assert calls == 1


@pytest.mark.parametrize(
    ("augmentation_factor", "added_raw_subjects", "message"),
    [
        (2, 0, "incomplete variants"),
        (1, 1, "Manifest fingerprint does not match"),
    ],
    ids=["changed-augmentation-factor", "changed-raw-manifest"],
)
def test_resume_config_rejects_stale_checkpoint(
    tmp_path: Path, augmentation_factor: int, added_raw_subjects: int, message: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    raw_manifest = write_raw_manifest(dataset_dir, raw_subjects(1))
    write_preprocessed_manifest(
        dataset_dir,
        preprocessed_subjects(1, tmp_path / "outputs"),
        status=ManifestStatus.RUNNING,
        augmentation_factor=1,
        raw_manifest_fingerprint=content_fingerprint(raw_manifest),
    )
    write_raw_manifest(
        dataset_dir, raw_subjects(1 + added_raw_subjects), sources=raw_manifest.sources
    )

    with pytest.raises(ValueError, match=message):
        PreprocessConfig(
            dataset_dir=dataset_dir,
            augmentation_factor=augmentation_factor,
            resume=True,
        )
