from pathlib import Path

import numpy as np
import pytest

from experiments.inpainting import detectors, prepare
from experiments.inpainting.tests.support import (
    AFFINE,
    LESION,
    SHAPE,
    THROUGH_PLANE,
    VESSEL,
    base_volume,
    read_subjects,
    write_raw_dataset,
)
from microbleednet.core.datamodels import PreprocessOutput
from microbleednet.core.engines import processor
from microbleednet.core.transforms import inpaint_vessels
from microbleednet.orchestration.manifests import RawSubject

BOTH = ("axial", "frangi3d")


@pytest.fixture
def subject(tmp_path: Path) -> RawSubject:
    return read_subjects(write_raw_dataset(tmp_path, ["a"]))[0]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_vessel_inpainting_is_captured_instead_of_applied(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepared = prepare.prepare_subject(subject, "QSM", "n4", BOTH, cache)

    np.testing.assert_array_equal(prepared.volume, base_volume())
    assert fake_pipeline[0].bias_field_correction == "n4"
    assert fake_pipeline[0].modality == "QSM"


def test_inpainting_is_restored_after_capturing(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    original = inpaint_vessels.apply

    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert processor.inpaint_vessels.apply is original


def test_lesions_brain_and_geometry_are_prepared(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepared = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert prepared.lesions.sum() == 1 and prepared.lesions[LESION]
    assert prepared.brain.all()
    np.testing.assert_array_equal(prepared.affine, AFFINE)
    assert prepared.spacing == (1.0, 1.0, 3.0)


def test_every_requested_strategy_gets_its_own_undilated_mask(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepared = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert set(prepared.vessels) == set(BOTH)
    assert prepared.vessels["axial"].sum() == 1 and prepared.vessels["axial"][VESSEL]
    assert prepared.vessels["frangi3d"].sum() == 2
    assert prepared.vessels["frangi3d"][THROUGH_PLANE]


def test_only_requested_strategies_are_returned(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepared = prepare.prepare_subject(subject, "QSM", "fast", ("axial",), cache)

    assert set(prepared.vessels) == {"axial"}


def test_a_cached_subject_skips_preprocessing(
    subject: RawSubject, cache: Path, fake_pipeline, monkeypatch
) -> None:
    first = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)
    monkeypatch.setattr(
        processor, "preprocess", lambda _: pytest.fail("preprocessing reran")
    )

    second = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    np.testing.assert_array_equal(second.volume, first.volume)
    np.testing.assert_array_equal(second.vessels["frangi3d"], first.vessels["frangi3d"])
    np.testing.assert_array_equal(second.affine, first.affine)


def test_a_cached_subject_does_not_detect_again(
    subject: RawSubject, cache: Path, fake_pipeline, monkeypatch
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)
    for name in BOTH:
        monkeypatch.setitem(
            detectors.DETECTORS, name, lambda _: pytest.fail("detected again")
        )

    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)


def test_a_new_strategy_is_added_to_an_existing_cache(
    subject: RawSubject, cache: Path, fake_pipeline, monkeypatch
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", ("axial",), cache)
    monkeypatch.setattr(
        processor, "preprocess", lambda _: pytest.fail("preprocessing reran")
    )
    monkeypatch.setitem(
        detectors.DETECTORS, "axial", lambda _: pytest.fail("axial detected again")
    )

    prepared = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert prepared.vessels["frangi3d"][THROUGH_PLANE]
    with np.load(cache / "fast" / "a.npz") as saved:
        assert {"vessels_axial", "vessels_frangi3d"} <= set(saved.files)


def test_the_cache_is_separate_per_bias_field_method(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)
    prepare.prepare_subject(subject, "QSM", "n4", BOTH, cache)

    assert len(fake_pipeline) == 2


def test_a_cache_without_the_affine_is_rebuilt(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    path = cache / "fast" / "a.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, volume=base_volume(), lesions=np.zeros(SHAPE), brain=np.ones(SHAPE)
    )

    prepared = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert len(fake_pipeline) == 1
    np.testing.assert_array_equal(prepared.affine, AFFINE)


def test_a_subject_without_a_mask_is_rejected(
    subject: RawSubject, cache: Path, monkeypatch
) -> None:
    def preprocess(preprocess_input):
        return PreprocessOutput(
            volume=base_volume(),
            mask=None,
            brain_mask=np.ones(SHAPE, dtype=np.uint8),
            affine=np.eye(4),
            bounding_box=((0, 1), (0, 1), (0, 1)),
            original_volume=preprocess_input.volume,
        )

    monkeypatch.setattr(processor, "preprocess", preprocess)

    with pytest.raises(AssertionError, match="lesion masks"):
        prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)


def test_the_cache_is_written_without_leaving_partial_files(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    assert [path.name for path in (cache / "fast").iterdir()] == ["a.npz"]


def test_a_finished_cache_entry_loads_without_any_computation(
    subject: RawSubject, cache: Path, fake_pipeline, monkeypatch
) -> None:
    first = prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)
    monkeypatch.setattr(
        processor, "preprocess", lambda _: pytest.fail("preprocessing reran")
    )

    loaded = prepare.load_prepared(cache / "fast" / "a.npz", BOTH)

    assert loaded is not None
    np.testing.assert_array_equal(loaded.volume, first.volume)
    np.testing.assert_array_equal(loaded.vessels["frangi3d"], first.vessels["frangi3d"])


def test_loading_only_returns_the_requested_strategies(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", BOTH, cache)

    loaded = prepare.load_prepared(cache / "fast" / "a.npz", ("axial",))

    assert loaded is not None and set(loaded.vessels) == {"axial"}


def test_a_cache_entry_without_a_requested_strategy_does_not_load(
    subject: RawSubject, cache: Path, fake_pipeline
) -> None:
    prepare.prepare_subject(subject, "QSM", "fast", ("axial",), cache)

    assert prepare.load_prepared(cache / "fast" / "a.npz", BOTH) is None


@pytest.mark.parametrize("content", [b"not an archive", b""])
def test_an_unreadable_cache_entry_does_not_load(cache: Path, content: bytes) -> None:
    path = cache / "fast" / "a.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    assert prepare.load_prepared(path, BOTH) is None


def test_a_missing_cache_entry_does_not_load(cache: Path) -> None:
    assert prepare.load_prepared(cache / "fast" / "a.npz", BOTH) is None
