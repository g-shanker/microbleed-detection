import csv
import io
import logging
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet import frst_report
from microbleednet.errors import ApplicationError
from microbleednet.frst_report import (
    SubjectScores,
    auc,
    froc,
    inspect_dataset,
    main,
    resolve_variants,
    score_map,
    write_csv,
    write_summary,
)
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)

SHAPE = (32, 32, 8)
LESIONS = ((10, 10, 4), (22, 20, 3))


def _save(path: Path, values: np.ndarray, spacing: float = 2.0) -> str:
    affine = np.diag([0.5, 0.5, spacing, 1.0])
    nib.save(nib.Nifti1Image(values.astype(np.float32), affine), path)
    return str(path)


def _subject(files: Path, subject_id: str, mask: np.ndarray | None):
    rng = np.random.default_rng(len(subject_id))
    volume = rng.normal(0.4, 0.02, SHAPE)
    brain = np.zeros(SHAPE)
    brain[3:29, 3:29, 1:7] = 1
    volume[brain == 0] = 0.0
    for x, y, z in LESIONS:
        volume[x - 1 : x + 2, y - 1 : y + 2, z] = 0.9
    volume[16, 4:28, 2] = 0.8
    volume_path = _save(files / f"{subject_id}.nii.gz", volume)
    return PreprocessedSubject(
        subject_id=subject_id,
        original_volume_path=volume_path,
        brain_mask_path=_save(files / f"{subject_id}_brain.nii.gz", brain),
        bounding_box=((0, 32), (0, 32), (0, 8)),
        variants=[
            PreprocessedVariant(
                volume_path=volume_path,
                frst_path=volume_path,
                mask_path=(
                    None
                    if mask is None
                    else _save(files / f"{subject_id}_mask.nii.gz", mask)
                ),
            )
        ],
    )


def _write_dataset(tmp_path: Path, masks: bool = True) -> Path:
    files = tmp_path / "files"
    files.mkdir()
    lesions = np.zeros(SHAPE)
    for center in LESIONS:
        lesions[center] = 1
    subjects = [
        _subject(files, "lesions", lesions if masks else None),
        _subject(files, "empty", np.zeros(SHAPE) if masks else None),
        _subject(files, "maskless", None),
    ]
    dataset_dir = tmp_path / "dataset"
    PreprocessedDatasetManifest(
        status=ManifestStatus.RUNNING,
        subjects=subjects,
        augmentation_factor=1,
        bias_field_correction="fast",
        raw_manifest_fingerprint="fingerprint",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    return dataset_dir


def test_inspect_dataset_scores_lesions_and_background(
    tmp_path: Path, caplog
) -> None:
    variants = resolve_variants(["sigma_0.25"], ["sigma_0.25+alpha_1"])

    with caplog.at_level(logging.INFO, logger="microbleednet.frst_report"):
        scores = inspect_dataset(_write_dataset(tmp_path), variants)

    assert "Scoring 3 FRST variants on 2 subjects" in caplog.text
    assert "lesions: variant 3/3 (sigma_0.25+alpha_1)" in caplog.text
    assert "[2/2] empty: done in 0m" in caplog.text
    assert "about 0m00s remaining" in caplog.text

    assert [item.variant for item in scores] == [
        "baseline",
        "sigma_0.25",
        "sigma_0.25+alpha_1",
    ] * 2
    lesions = [item for item in scores if item.subject_id == "lesions"]
    empty = [item for item in scores if item.subject_id == "empty"]
    assert all(item.lesion_scores.size == 2 for item in lesions)
    assert all(item.lesion_scores.size == 0 for item in empty)
    assert all(item.background_peaks.size > 0 for item in scores)
    assert lesions[0].spacing == (0.5, 0.5, 2.0)


def test_inspect_dataset_requires_masked_subjects(tmp_path: Path) -> None:
    with pytest.raises(ApplicationError, match="subjects with masks"):
        inspect_dataset(_write_dataset(tmp_path, masks=False), resolve_variants([], []))


def test_resolve_variants_rejects_unknown_names() -> None:
    with pytest.raises(ApplicationError, match="Unknown FRST variant"):
        resolve_variants(["missing"], [])


def test_froc_counts_lesions_above_allowed_background() -> None:
    lesions = np.array([0.9, 0.5, 0.1])
    background = np.array([0.8, 0.6, 0.4, 0.2])

    # 1 subject: 1, 5, 10, and 20 allowed false positives.
    assert froc(lesions, background, 1) == pytest.approx((1 / 3, 1.0, 1.0, 1.0))
    assert all(math.isnan(value) for value in froc(np.array([]), background, 1))


def test_auc_ranks_lesions_against_background() -> None:
    assert auc(np.array([1.0, 0.5]), np.array([0.5, 0.0])) == pytest.approx(0.875)
    assert math.isnan(auc(np.array([1.0]), np.array([])))


def test_score_map_without_background_region_has_no_peaks() -> None:
    frst_map = np.random.default_rng(0).random((8, 8, 4))

    lesions, background = score_map(
        frst_map, [(4, 4, 2)], np.zeros(frst_map.shape, dtype=bool), (1.0, 1.0, 1.0)
    )

    assert lesions.shape == (1,)
    assert background.size == 0


def _scores(variant: str, lesion: float) -> SubjectScores:
    return SubjectScores(
        "subject",
        variant,
        (0.5, 0.5, 2.0),
        np.array([lesion]),
        np.linspace(0.0, 0.85, 30),
    )


def test_write_summary_suggests_combining_improvements() -> None:
    output = io.StringIO()

    write_summary(
        [_scores("baseline", 0.4), _scores("alpha_1", 0.9), _scores("a+b", 0.9)],
        output,
    )

    text = output.getvalue()
    assert "voxel spacing (mm): 0.50x0.50x2.00 (1)" in text
    assert "next: rerun with --combine alpha_1" in text


def test_write_summary_reports_no_improvement() -> None:
    output = io.StringIO()

    write_summary([_scores("baseline", 0.9), _scores("alpha_1", 0.4)], output)

    assert "no single variant beats baseline" in output.getvalue()


def test_main_writes_csv_and_summary(tmp_path: Path, capsys) -> None:
    dataset_dir = _write_dataset(tmp_path)

    main([str(dataset_dir), "--variants", "alpha_1", "--subjects", "1"])

    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert [row["variant"] for row in rows] == ["baseline", "alpha_1"]
    assert rows[0]["spacing_mm"] == "0.50x0.50x2.00"
    assert "sens@10" in captured.err


def test_inspect_dataset_resumes_from_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = _write_dataset(tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    variants = resolve_variants(["alpha_1"], [])
    first = io.StringIO()
    write_csv(inspect_dataset(dataset_dir, variants, checkpoint=checkpoint), first)

    def fail(*args, **kwargs):
        raise AssertionError("subject should come from the checkpoint")

    monkeypatch.setattr(frst_report.frst_variants, "compute", fail)
    resumed = io.StringIO()
    write_csv(inspect_dataset(dataset_dir, variants, checkpoint=checkpoint), resumed)

    assert resumed.getvalue() == first.getvalue()
