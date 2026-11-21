import csv
import io
import logging
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet import inpainting_report
from microbleednet.core.datamodels import PreprocessOutput
from microbleednet.errors import ApplicationError
from microbleednet.inpainting_report import (
    SettingSummary,
    format_duration,
    inspect_dataset,
    inspect_subject,
    main,
    recommend,
    slice_vesselness,
    summarize,
    write_csv,
)
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    timestamp,
)

SHAPE = (20, 20, 8)
IN_PLANE_LESION = (14, 12, 4)
OUT_OF_PLANE_LESION = (8, 10, 6)
DARK_LESION = (15, 4, 2)


def _prepared(with_lesions: bool = True) -> PreprocessOutput:
    rng = np.random.default_rng(0)
    volume = 0.3 + rng.normal(0.0, 0.01, SHAPE)
    brain = np.ones(SHAPE, dtype=np.uint8)
    brain[:, :, 0] = 0
    volume[:, :, 0] = 0.0
    volume[:, 9:12, 4] = 0.7
    volume[:, 10, 4] = 1.0
    mask = np.zeros(SHAPE, dtype=np.uint8)
    if with_lesions:
        for center in ((4, 4, 4), IN_PLANE_LESION, OUT_OF_PLANE_LESION):
            volume[center] = 1.0
            mask[center] = 1
        # A darker-than-tissue annotation has no usable contrast ratio.
        volume[13:18, 2:7, 1:4] = 0.1
        mask[DARK_LESION] = 1
    return PreprocessOutput(
        volume=volume,
        mask=mask,
        brain_mask=brain,
        affine=np.eye(4),
        bounding_box=((0, 20), (0, 20), (0, 8)),
        original_volume=nib.Nifti1Image(volume, np.eye(4)),
    )


def _vessel_line(volume: np.ndarray) -> np.ndarray:
    vessels = np.zeros(volume.shape, dtype=bool)
    vessels[:, 10, 4] = True
    return vessels


def _write_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    files = tmp_path / "files"
    files.mkdir()
    image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    nib.save(image, files / "volume.nii.gz")
    now = timestamp()
    RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        sources=[
            RawSource(
                input_dir=str(files),
                mask_dir=str(files),
                volume_pattern="{subject_id}.nii.gz",
                mask_pattern="{subject_id}_mask.nii.gz",
                source_id="source",
                modality="T2*-GRE",
                added_on=now,
            )
        ],
        subjects=[
            RawSubject(
                subject_id=subject_id,
                source_id="source",
                volume_path=str(files / "volume.nii.gz"),
                mask_path=None if subject_id == "maskless" else str(
                    files / "volume.nii.gz"
                ),
            )
            for subject_id in ("first", "second", "maskless")
        ],
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    return dataset_dir


@pytest.fixture
def stub_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        inpainting_report.processor,
        "prepare_volume",
        lambda preprocess_input: _prepared(),
    )
    monkeypatch.setattr(
        inpainting_report.inpaint_vessels, "get_volume_vessel_mask", _vessel_line
    )


def _column(records, mode: str, name: str) -> list:
    return [getattr(item, name) for item in records if item.mode == mode]


def test_inspect_dataset_compares_3d_and_in_plane_dilation(
    tmp_path: Path, stub_pipeline, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger=inpainting_report.__name__):
        records = inspect_dataset(_write_dataset(tmp_path), "n4", max_iterations=2)

    assert "Inspecting 2 subjects, 5 dilation settings each" in caplog.text
    assert "first: setting 5/5 (in_plane, k=2)" in caplog.text
    assert "[2/2] second: done in 0m" in caplog.text
    assert "about 0m00s remaining" in caplog.text

    assert {item.subject_id for item in records} == {"first", "second"}
    first = [item for item in records if item.subject_id == "first"]
    assert _column(first, "3d", "iterations") == [0, 1, 2]
    assert _column(first, "3d", "lesions_touched") == [0, 0, 2]
    assert _column(first, "in_plane", "lesions_touched") == [0, 0, 1]
    fractions = _column(first, "3d", "inpainted_brain_fraction")
    assert fractions[0] < fractions[1] < fractions[2]
    in_plane = _column(first, "in_plane", "inpainted_brain_fraction")
    assert in_plane[2] < fractions[2]
    vesselness = _column(first, "in_plane", "residual_vesselness")
    assert vesselness[1] < vesselness[0]
    assert all(item.lesions == 4 for item in first)
    assert all(len(item.contrast_ratios) == 3 for item in first)
    assert all(item.spurious_frst_components >= 0 for item in first)


def test_slice_vesselness_handles_constant_and_structured_slices() -> None:
    image_slice = np.zeros((16, 16))
    assert not slice_vesselness(image_slice).any()

    image_slice[:, 8] = 1.0
    assert slice_vesselness(image_slice).shape == (16, 16)


def test_subject_without_vessels_or_lesions_recommends_no_dilation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        inpainting_report.inpaint_vessels,
        "get_volume_vessel_mask",
        lambda volume: np.zeros(volume.shape, dtype=bool),
    )

    records = inspect_subject("empty", _prepared(with_lesions=False), 1, 0.2)
    summaries = summarize(records, "3d")

    assert all(math.isnan(item.residual_vesselness) for item in records)
    assert all(item.touched_fraction == 0.0 for item in summaries)
    assert math.isnan(summaries[0].median_contrast_ratio)
    assert recommend(summaries) == (0, "dilation does not reduce residual vesselness")


def _summary(iterations: int, vesselness: float, touched: float) -> SettingSummary:
    return SettingSummary(
        mode="3d",
        iterations=iterations,
        inpainted_brain_fraction=0.01 * iterations,
        touched_fraction=touched,
        median_contrast_ratio=1.0,
        residual_vesselness=vesselness,
        spurious_frst_components=0.0,
    )


def test_recommend_picks_smallest_count_near_full_reduction() -> None:
    summaries = [_summary(0, 1.0, 0.0), _summary(1, 0.05, 0.0), _summary(2, 0.0, 0.0)]

    iterations, reason = recommend(summaries)

    assert iterations == 1
    assert "smallest count" in reason


def test_recommend_respects_lesion_constraint() -> None:
    summaries = [_summary(0, 1.0, 0.0), _summary(1, 0.0, 0.5), _summary(2, 0.5, 0.0)]

    iterations, reason = recommend(summaries)

    assert iterations == 2
    assert "largest reduction (50%)" in reason


def test_inspect_dataset_requires_masked_subjects(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)
    manifest_path = DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    manifest = RawDatasetManifest.read(manifest_path)
    manifest.model_copy(
        update={
            "subjects": [
                subject.model_copy(update={"mask_path": None})
                for subject in manifest.subjects
            ]
        }
    ).write(manifest_path)

    with pytest.raises(ApplicationError, match="subjects with masks"):
        inspect_dataset(dataset_dir, "fast")


def test_main_writes_csv_and_recommendations(
    tmp_path: Path, stub_pipeline, capsys
) -> None:
    dataset_dir = _write_dataset(tmp_path)

    main(
        [
            str(dataset_dir),
            "--bias-field-correction",
            "fast",
            "--max-iterations",
            "1",
            "--subjects",
            "1",
        ]
    )

    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert len(rows) == 4
    assert {row["subject_id"] for row in rows} < {"first", "second"}
    assert "contrast_ratios" not in rows[0]
    assert captured.err.count("recommended k=") == 2
    assert "overall: mode=" in captured.err


def test_main_rejects_zero_max_iterations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [str(tmp_path), "--bias-field-correction", "fast", "--max-iterations", "0"]
        )


def test_format_duration_includes_hours_only_when_needed() -> None:
    assert format_duration(75.4) == "1m15s"
    assert format_duration(3725) == "1h02m05s"


def test_inspect_dataset_resumes_from_checkpoint(
    tmp_path: Path, stub_pipeline, monkeypatch, caplog
) -> None:
    dataset_dir = _write_dataset(tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = io.StringIO()
    write_csv(
        inspect_dataset(dataset_dir, "n4", max_iterations=1, checkpoint=checkpoint),
        first,
    )

    def fail(preprocess_input):
        raise AssertionError("subject should come from the checkpoint")

    monkeypatch.setattr(inpainting_report.processor, "prepare_volume", fail)
    resumed = io.StringIO()
    with caplog.at_level(logging.INFO, logger=inpainting_report.__name__):
        write_csv(
            inspect_dataset(
                dataset_dir, "n4", max_iterations=1, checkpoint=checkpoint
            ),
            resumed,
        )

    assert resumed.getvalue() == first.getvalue()
    assert "2 already in checkpoint" in caplog.text
    assert "[2/2] second: loaded from checkpoint" in caplog.text
