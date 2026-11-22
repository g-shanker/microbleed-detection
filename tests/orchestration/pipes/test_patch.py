from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from microbleednet.core import utils
from microbleednet.core.datamodels import ExtractedPatches
from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
)
from microbleednet.orchestration.layouts import ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PatchManifest,
    PreprocessedSubject,
)
from microbleednet.orchestration.pipes import patch
from tests.support import preprocessed_subjects
from tests.support_pipes import StubDetector

PATCHES_PER_VARIANT = 8  # a 2x2x2 volume split into 1x1x1 patches
MICROBLEED_VOXEL_MASK = np.zeros((2, 2, 2), dtype=np.uint8)
MICROBLEED_VOXEL_MASK[0, 0, 0] = 1


@pytest.fixture(autouse=True)
def no_gpu_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch, "release_gpu_memory", lambda: None)


@pytest.fixture
def layout(tmp_path: Path) -> ExperimentLayout:
    return ExperimentLayout(experiment_dir=tmp_path / "experiment")


@pytest.fixture(scope="module")
def subjects(tmp_path_factory: pytest.TempPathFactory) -> list[PreprocessedSubject]:
    """Two read-only subjects with two variants each; tests must not mutate files."""
    return preprocessed_subjects(
        2,
        tmp_path_factory.mktemp("inputs"),
        variants=2,
        write_files=True,
        data=MICROBLEED_VOXEL_MASK,
    )


def non_overlapping(
    layout: ExperimentLayout,
    subjects: list[PreprocessedSubject],
    **overrides: Any,
) -> NonOverlappingPatchConfig:
    values: dict[str, Any] = {
        "experiment_layout": layout,
        "stage": "detector",
        "split": "train",
        "subjects": subjects,
        "patch_size": 1,
        "augmentation_factor": 2,
    }
    values.update(overrides)
    return NonOverlappingPatchConfig(**values)


def read_manifest(config: BasePatchConfig) -> PatchManifest:
    return PatchManifest.read(
        config.experiment_layout.patch_manifest_path(config.stage, config.split)
    )


@pytest.fixture
def config(
    layout: ExperimentLayout, subjects: list[PreprocessedSubject]
) -> NonOverlappingPatchConfig:
    return non_overlapping(layout, subjects[:1])


@pytest.fixture(scope="module")
def manifest(
    subjects: list[PreprocessedSubject], tmp_path_factory: pytest.TempPathFactory
) -> PatchManifest:
    """Manifest of one non-overlapping extraction, shared by read-only tests."""
    layout = ExperimentLayout(experiment_dir=tmp_path_factory.mktemp("experiment"))
    config = non_overlapping(layout, subjects[:1])
    patch.execute(config)
    return read_manifest(config)


def test_execute_writes_complete_manifest_describing_config(
    manifest: PatchManifest,
) -> None:
    assert manifest.status is ManifestStatus.COMPLETE
    assert (manifest.stage, manifest.split) == ("detector", "train")
    assert manifest.subject_ids == ["subject-0"]
    assert manifest.patch_size == 1
    assert manifest.augmentation_factor == 2
    assert manifest.probability_threshold is None
    assert manifest.use_amp is False


@pytest.mark.parametrize(("factor", "variants"), [(1, 1), (2, 2)])
def test_execute_extracts_only_configured_variants(
    layout: ExperimentLayout,
    subjects: list[PreprocessedSubject],
    factor: int,
    variants: int,
) -> None:
    config = non_overlapping(layout, subjects[:1], augmentation_factor=factor)

    patch.execute(config)

    records = read_manifest(config).records
    assert len(records) == variants * PATCHES_PER_VARIANT
    assert len({record.volume_path for record in records}) == variants


def test_execute_numbers_patches_within_each_variant(
    manifest: PatchManifest,
) -> None:
    assert [record.patch_index for record in manifest.records] == (
        list(range(PATCHES_PER_VARIANT)) * 2
    )


def test_execute_labels_only_patches_containing_mask(manifest: PatchManifest) -> None:
    assert [record.has_microbleed for record in manifest.records] == (
        [True] + [False] * (PATCHES_PER_VARIANT - 1)
    ) * 2


@pytest.mark.parametrize("field", ["volume_path", "mask_path", "frst_path"])
def test_execute_saves_float32_patch_stacks(
    manifest: PatchManifest, field: str
) -> None:
    array = np.load(getattr(manifest.records[0], field))

    assert array.shape == (PATCHES_PER_VARIANT, 1, 1, 1)
    assert array.dtype == np.float32


def test_execute_only_extracts_supplied_subjects(manifest: PatchManifest) -> None:
    patch_dir = Path(manifest.records[0].volume_path).parent
    names = [path.name for path in patch_dir.glob("*.npy")]

    assert names
    assert all("subject-0" in name for name in names)


def test_execute_rejects_unsupported_patch_config() -> None:
    with pytest.raises(TypeError, match="supported patch configuration"):
        patch.execute(cast(Any, object()))


def drop_later_subject_mask(
    subjects: list[PreprocessedSubject],
) -> list[PreprocessedSubject]:
    unmasked = subjects[1].variants[0].model_copy(update={"mask_path": None})
    return [subjects[0], subjects[1].model_copy(update={"variants": [unmasked]})]


def point_frst_at_missing_file(
    subjects: list[PreprocessedSubject],
) -> list[PreprocessedSubject]:
    first = subjects[0].variants[0]
    missing = first.model_copy(
        update={"frst_path": str(Path(first.frst_path).with_name("missing.nii.gz"))}
    )
    return [subjects[0].model_copy(update={"variants": [missing]}), *subjects[1:]]


@pytest.mark.parametrize(
    ("mutate", "factor", "message"),
    [
        (drop_later_subject_mask, 1, "requires a variant mask"),
        (point_frst_at_missing_file, 1, "patch input file is missing"),
        (lambda subjects: subjects, 3, "augmentation factor exceeds"),
    ],
    ids=["later-subject-without-mask", "missing-input-file", "too-many-variants"],
)
def test_config_rejects_unextractable_subjects_before_any_output(
    layout: ExperimentLayout,
    subjects: list[PreprocessedSubject],
    mutate: Callable[[list[PreprocessedSubject]], list[PreprocessedSubject]],
    factor: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        non_overlapping(layout, mutate(subjects), augmentation_factor=factor)

    assert not layout.patch_dir_path("detector", "train").exists()
    assert not layout.patch_manifest_path("detector", "train").exists()


def test_execute_rejects_subjects_without_extracted_patches(
    monkeypatch: pytest.MonkeyPatch, config: NonOverlappingPatchConfig
) -> None:
    empty = np.empty((0, 1, 1, 1))
    monkeypatch.setattr(
        patch,
        "NonOverlappingExtractor",
        lambda size: lambda volume, mask, frst: ExtractedPatches(empty, empty, empty),
    )

    with pytest.raises(ApplicationError, match="Patch extraction produced no records"):
        patch.execute(config)

    assert not config.experiment_layout.patch_manifest_path(
        "detector", "train"
    ).exists()


class FakeExtractor:
    """Logs construction and calls on ``run``; returns each volume as one patch."""

    def __init__(self, run: SimpleNamespace, kwargs: dict[str, Any]) -> None:
        self.run = run
        run.instances.append(kwargs)

    def __call__(
        self, volume: np.ndarray, mask: np.ndarray, frst: np.ndarray
    ) -> ExtractedPatches:
        self.run.calls += 1
        return ExtractedPatches(volumes=volume[None], masks=mask[None], frst=frst[None])


@pytest.fixture(scope="module")
def target_run(
    subjects: list[PreprocessedSubject], tmp_path_factory: pytest.TempPathFactory
) -> SimpleNamespace:
    """One target-centered extraction with a fake extractor, for read-only tests."""
    config = TargetCenteredPatchConfig(
        experiment_layout=ExperimentLayout(
            experiment_dir=tmp_path_factory.mktemp("target")
        ),
        stage="student",
        split="train",
        subjects=subjects,
        patch_size=2,
        augmentation_factor=1,
        probability_threshold=0.5,
        detector=StubDetector(),
        use_amp=True,
    )
    run = SimpleNamespace(config=config, instances=[], calls=0)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            patch,
            "TargetCenteredExtractor",
            lambda **kwargs: FakeExtractor(run, kwargs),
        )
        patch.execute(config)
    run.manifest = read_manifest(config)
    return run


def test_target_centered_creates_one_extractor_from_config(
    target_run: SimpleNamespace,
) -> None:
    assert target_run.instances == [
        {
            "detector": target_run.config.detector,
            "threshold": 0.5,
            "patch_size": 2,
            "use_amp": True,
        }
    ]


def test_target_centered_reuses_extractor_for_all_subjects(
    target_run: SimpleNamespace,
) -> None:
    assert target_run.calls == len(target_run.config.subjects)


def test_target_centered_manifest_records_threshold_and_amp(
    target_run: SimpleNamespace,
) -> None:
    manifest = target_run.manifest
    assert manifest.subject_ids == ["subject-0", "subject-1"]
    assert manifest.probability_threshold == 0.5
    assert manifest.use_amp is True
    assert len(manifest.records) == 2


def test_manifest_matches_the_config_that_produced_it(
    target_run: SimpleNamespace,
) -> None:
    assert patch.manifest_matches_config(target_run.manifest, target_run.config)


@pytest.mark.parametrize(
    "update",
    [
        {"use_amp": False},
        {"probability_threshold": 0.9},
        {"patch_size": 4},
        {"augmentation_factor": 2},
        {"stage": "teacher"},
        {"split": "validation"},
        {"subjects": []},
    ],
    ids=lambda update: next(iter(update)),
)
def test_manifest_rejects_changed_config(
    target_run: SimpleNamespace, update: dict[str, Any]
) -> None:
    changed = target_run.config.model_copy(update=update)

    assert not patch.manifest_matches_config(target_run.manifest, changed)


def test_manifest_rejects_running_status(target_run: SimpleNamespace) -> None:
    running = target_run.manifest.model_copy(update={"status": ManifestStatus.RUNNING})

    assert not patch.manifest_matches_config(running, target_run.config)


def test_manifest_rejects_missing_record_arrays(
    tmp_path: Path, target_run: SimpleNamespace
) -> None:
    stale = target_run.manifest.records[0].model_copy(
        update={"mask_path": str(tmp_path / "missing.npy")}
    )
    manifest = target_run.manifest.model_copy(update={"records": [stale]})

    assert not patch.manifest_matches_config(manifest, target_run.config)


def make_extractor(
    monkeypatch: pytest.MonkeyPatch, *, logits: torch.Tensor, **kwargs: Any
) -> patch.TargetCenteredExtractor:
    monkeypatch.setattr(utils, "predict_logits", lambda model, volume, use_amp: logits)
    return patch.TargetCenteredExtractor(
        detector=cast(Any, object()), use_amp=False, **kwargs
    )


def test_extractor_returns_empty_patches_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = make_extractor(
        monkeypatch, logits=torch.zeros((2, 2, 2, 2)), threshold=0.9, patch_size=24
    )
    volume = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    frst = np.ones((2, 2, 2), dtype=np.float64)

    extracted = extractor(volume, mask, frst)

    assert [
        array.shape for array in (extracted.volumes, extracted.masks, extracted.frst)
    ] == [(0, 24, 24, 24)] * 3
    assert (extracted.volumes.dtype, extracted.masks.dtype, extracted.frst.dtype) == (
        volume.dtype,
        mask.dtype,
        frst.dtype,
    )


def test_extractor_centers_patch_on_each_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((2, 2, 2, 2))
    logits[1, 0, 0, 0] = 10
    extractor = make_extractor(monkeypatch, logits=logits, threshold=0.5, patch_size=2)

    extracted = extractor(
        np.ones((2, 2, 2)), np.zeros((2, 2, 2)), np.full((2, 2, 2), 2.0)
    )

    centered_volume = np.zeros((1, 2, 2, 2))
    centered_volume[0, 1, 1, 1] = 1
    np.testing.assert_array_equal(extracted.volumes, centered_volume)
    np.testing.assert_array_equal(extracted.masks, np.zeros((1, 2, 2, 2)))
    np.testing.assert_array_equal(extracted.frst, centered_volume * 2)
