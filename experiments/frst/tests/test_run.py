import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from experiments.frst import run
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)

SHAPE = (24, 24, 4)


def save(path: Path, array: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return str(path)


def make_subject(directory: Path, subject_id: str, *, with_mask: bool = True):
    x, y = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    disc = ((x - 12) ** 2 + (y - 12) ** 2 <= 9).astype(np.float32)
    volume = np.broadcast_to(disc[:, :, None], SHAPE).astype(np.float32)
    lesion = np.zeros(SHAPE, dtype=np.uint8)
    lesion[12, 12, :] = 1
    volume_path = save(directory / f"{subject_id}-volume.nii.gz", volume)
    return PreprocessedSubject(
        subject_id=subject_id,
        original_volume_path=volume_path,
        brain_mask_path=save(
            directory / f"{subject_id}-brain.nii.gz", np.ones(SHAPE, dtype=np.uint8)
        ),
        bounding_box=((0, 1), (0, 1), (0, 1)),
        variants=[
            PreprocessedVariant(
                volume_path=volume_path,
                mask_path=(
                    save(directory / f"{subject_id}-mask.nii.gz", lesion)
                    if with_mask
                    else None
                ),
                frst_path=volume_path,
            )
        ],
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    subjects = [
        make_subject(tmp_path, "a"),
        make_subject(tmp_path, "b"),
        make_subject(tmp_path, "unmasked", with_mask=False),
    ]
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=subjects,
        bias_field_correction="fast",
        raw_manifest_fingerprint="raw",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    return dataset_dir


def write_config(tmp_path: Path, dataset: Path, extra: str = "") -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        f'dataset_dir = "{dataset.as_posix()}"\n'
        f'output_dir = "{(tmp_path / "results").as_posix()}"\n'
        "peak_min_distance = 2\nrecall_at = [1]\n" + extra
    )
    return path


def test_select_subjects_skips_unmasked_subjects(dataset: Path) -> None:
    manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset).preprocessed_manifest_path()
    )

    selected = run.select_subjects(manifest.subjects, None, seed=0)

    assert [subject.subject_id for subject in selected] == ["a", "b"]


def test_select_subjects_takes_a_seeded_subset(dataset: Path) -> None:
    manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset).preprocessed_manifest_path()
    )

    first = run.select_subjects(manifest.subjects, 1, seed=3)
    second = run.select_subjects(manifest.subjects, 1, seed=3)

    assert len(first) == 1
    assert first == second


def test_main_writes_ranked_results_and_variant_specs(
    tmp_path: Path, dataset: Path, capsys
) -> None:
    config = write_config(tmp_path, dataset, "[sweep]\nalpha = [3]\n")

    assert run.main(["--config", str(config)]) == 0

    with (tmp_path / "results" / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert {row["name"] for row in rows} == {"pipeline", "alpha=3"}
    assert all(row["subjects"] == "2" and row["lesions"] == "2.0" for row in rows)
    assert float(rows[0]["mean_auc"]) >= float(rows[1]["mean_auc"])
    specs = json.loads((tmp_path / "results" / "variants.json").read_text())
    assert specs["alpha=3"]["alpha"] == 3
    assert "mean_auc" in capsys.readouterr().out


def test_dry_run_lists_variants_without_computing(
    tmp_path: Path, dataset: Path, capsys
) -> None:
    config = write_config(tmp_path, dataset, "[sweep]\nalpha = [3]\n")

    assert run.main(["--config", str(config), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "pipeline" in output and "alpha=3" in output
    assert not (tmp_path / "results").exists()


def test_main_fails_when_no_subject_has_a_mask(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=[make_subject(tmp_path, "x", with_mask=False)],
        bias_field_correction="fast",
        raw_manifest_fingerprint="raw",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())

    assert run.main(["--config", str(write_config(tmp_path, dataset_dir))]) == 1
