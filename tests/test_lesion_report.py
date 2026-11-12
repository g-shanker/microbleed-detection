import csv
import io
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet.errors import ApplicationError
from microbleednet.lesion_report import (
    frst_slice_axis,
    frst_slice_evidence,
    inspect_run,
    local_contrast,
    main,
    write_summary,
)
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    InferManifest,
    InferredSubject,
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    timestamp,
)

SHAPE = (24, 24, 24)
DETECTED = (12, 12, 12)
MISSED = (6, 17, 12)


def save(path: Path, values: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(values.astype(np.float32), np.eye(4)), path)
    return str(path)


def write_run(
    tmp_path: Path,
    *,
    detections: bool = True,
    with_mask: bool = True,
    inference_status: ManifestStatus = ManifestStatus.COMPLETE,
    inferred_subject_id: str = "subject",
    probability_shape: tuple[int, int, int] = SHAPE,
) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    files = tmp_path / "files"
    files.mkdir()
    rng = np.random.default_rng(0)

    raw = rng.normal(100.0, 5.0, SHAPE)
    reference = np.zeros(SHAPE)
    for center in (DETECTED, MISSED):
        raw[center] = 40.0
        reference[center] = 1
    reference[DETECTED[0], DETECTED[1], DETECTED[2] + 1] = 1
    preprocessed = 200.0 - raw
    brain = np.ones(SHAPE)
    brain[:3] = 0
    frst = rng.uniform(0.0, 0.01, SHAPE)
    frst[3] = 1.0
    frst[DETECTED] = 1.0
    frst[MISSED] = 0.25

    probability = np.zeros(SHAPE)
    probability[DETECTED] = 0.9
    probability[MISSED[0] + 1, MISSED[1], MISSED[2]] = 0.05
    final = np.zeros(SHAPE)
    if detections:
        final[DETECTED] = 1

    raw_path = save(files / "raw.nii.gz", raw)
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
                subject_id="subject",
                source_id="source",
                volume_path=raw_path,
                mask_path=(
                    save(files / "mask.nii.gz", reference) if with_mask else None
                ),
            )
        ],
    ).write(DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path())
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=[
            PreprocessedSubject(
                subject_id="subject",
                original_volume_path=raw_path,
                brain_mask_path=save(files / "brain.nii.gz", brain),
                bounding_box=((0, 24), (0, 24), (0, 24)),
                variants=[
                    PreprocessedVariant(
                        volume_path=save(files / "pre.nii.gz", preprocessed),
                        frst_path=save(files / "frst.nii.gz", frst),
                    )
                ],
            )
        ],
        augmentation_factor=1,
        bias_field_correction="fast",
        raw_manifest_fingerprint="fingerprint",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    InferManifest(
        status=inference_status,
        device="cpu",
        detector_checkpoint_path="detector.pth",
        student_checkpoint_path="student.pth",
        detector_threshold=0.2,
        student_threshold=0.0,
        discriminator_patch_size=24,
        minimum_volume_mm3=0.0,
        maximum_ellipticity=1.0,
        minimum_brain_distance_mm=0.0,
        subjects=[
            InferredSubject(
                subject_id=inferred_subject_id,
                detector_output_path=save(
                    files / "probability.nii.gz",
                    np.resize(probability, probability_shape),
                ),
                discriminator_output_path=save(files / "final.nii.gz", final),
            )
        ],
    ).write(ExperimentLayout(experiment_dir=run_dir).inference_manifest_path())
    return dataset_dir, run_dir


def test_inspect_run_describes_detected_and_missed_lesions(tmp_path: Path) -> None:
    dataset_dir, run_dir = write_run(tmp_path)

    missed, detected = inspect_run(dataset_dir, run_dir)

    assert (detected.voxel_i, detected.voxel_j, detected.voxel_k) == DETECTED
    assert detected.annotation_voxels == 2
    assert detected.detected
    assert detected.nearest_detection_mm == 0.0
    assert detected.probability_at_point == pytest.approx(0.9)
    assert detected.max_probability_offset_mm == 0.0
    assert detected.in_brain_mask
    assert detected.brain_edge_distance_mm == pytest.approx(10.0)
    assert detected.world_x_mm == pytest.approx(12.0)
    assert detected.raw_contrast > 5
    assert detected.preprocessed_contrast == pytest.approx(detected.raw_contrast)
    assert detected.frst_contrast > 5
    assert detected.frst_slice_fraction == pytest.approx(1.0)
    assert detected.slice_peak_edge_distance_mm == pytest.approx(1.0)
    assert detected.slice_strong_frst_near_edge == pytest.approx(24 / 25)

    assert not missed.detected
    assert missed.annotation_voxels == 1
    assert missed.nearest_detection_mm == pytest.approx(math.dist(MISSED, DETECTED))
    assert missed.probability_at_point == 0.0
    assert missed.max_probability_nearby == pytest.approx(0.05)
    assert missed.max_probability_offset_mm == pytest.approx(1.0)
    assert missed.brain_edge_distance_mm == pytest.approx(4.0)
    assert missed.frst_slice_fraction == pytest.approx(0.25)


def test_inspect_run_handles_runs_without_detections(tmp_path: Path) -> None:
    dataset_dir, run_dir = write_run(tmp_path, detections=False)

    records = inspect_run(dataset_dir, run_dir)

    assert not any(record.detected for record in records)
    assert all(math.isnan(record.nearest_detection_mm) for record in records)
    summary = io.StringIO()
    write_summary(records, summary)
    assert "lesions=2 detected=0 missed=2" in summary.getvalue()
    assert "nan" in summary.getvalue()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"inference_status": ManifestStatus.RUNNING}, "incomplete inference"),
        ({"inferred_subject_id": "other"}, "missing from the dataset"),
        ({"with_mask": False}, "requires a reference mask"),
        ({"probability_shape": (24, 24, 23)}, "do not share one shape"),
    ],
)
def test_inspect_run_rejects_invalid_inputs(
    tmp_path: Path, options: dict, message: str
) -> None:
    dataset_dir, run_dir = write_run(tmp_path, **options)

    with pytest.raises(ApplicationError, match=message):
        inspect_run(dataset_dir, run_dir)


def test_frst_slice_axis_follows_canonical_superior_axis() -> None:
    assert frst_slice_axis(np.eye(4)) == 2
    first_axis_superior = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
    )
    assert frst_slice_axis(first_axis_superior) == 0


def test_frst_slice_evidence_is_undefined_for_empty_slice() -> None:
    evidence = frst_slice_evidence(
        np.zeros(SHAPE), (12, 12, 12), 2, np.ones(SHAPE), np.ones(3)
    )

    assert all(math.isnan(value) for value in evidence)


def test_local_contrast_is_undefined_without_usable_background() -> None:
    spacing = np.ones(3)
    volume = np.ones(SHAPE)

    assert math.isnan(
        local_contrast(volume, (12, 12, 12), spacing, np.zeros(SHAPE, bool), True)
    )
    assert math.isnan(
        local_contrast(volume, (12, 12, 12), spacing, np.ones(SHAPE, bool), True)
    )


def test_main_writes_csv_and_summary(tmp_path: Path, capsys) -> None:
    dataset_dir, run_dir = write_run(tmp_path)

    main([str(dataset_dir), str(run_dir), "--search-radius-mm", "2"])

    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert [row["detected"] for row in rows] == ["False", "True"]
    assert rows[0]["subject_id"] == "subject"
    assert "lesions=2 detected=1 missed=1" in captured.err
    assert "missed outside brain mask: 0/1" in captured.err
