import csv
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet.errors import ApplicationError
from microbleednet.intensity_report import inspect_manifest, main, write_csv
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)


def save_volume(path: Path, values: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    return str(path)


def write_manifest(
    tmp_path: Path,
    volume: np.ndarray,
    brain_mask: np.ndarray,
    *,
    variants: bool = True,
) -> Path:
    volume_path = save_volume(tmp_path / "volume.nii.gz", volume)
    brain_mask_path = save_volume(tmp_path / "brain_mask.nii.gz", brain_mask)
    variant = PreprocessedVariant(
        volume_path=volume_path,
        mask_path=None,
        frst_path=volume_path,
    )
    manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.RUNNING,
        subjects=[
            PreprocessedSubject(
                subject_id="subject-1",
                original_volume_path=volume_path,
                brain_mask_path=brain_mask_path,
                bounding_box=((0, 2), (0, 2), (0, 2)),
                variants=[variant] if variants else [],
            )
        ],
        augmentation_factor=1,
        bias_field_correction="fast",
        raw_manifest_fingerprint="fingerprint",
    )
    manifest_path = tmp_path / "preprocessed.json"
    manifest.write(manifest_path)
    return manifest_path


def test_inspect_manifest_reports_brain_masked_variant_zero(tmp_path: Path) -> None:
    volume = np.array(
        [[[0.0, -0.1], [0.5, 1.1]], [[np.nan, 0.7], [0.8, 0.9]]]
    )
    brain_mask = np.ones_like(volume)
    manifest_path = write_manifest(tmp_path, volume, brain_mask)

    statistics = inspect_manifest(manifest_path)

    assert len(statistics) == 1
    subject = statistics[0]
    assert subject.subject_id == "subject-1"
    assert subject.voxel_count == 8
    assert subject.minimum == pytest.approx(-0.1)
    assert subject.maximum == pytest.approx(1.1)
    assert subject.mean == pytest.approx(3.9 / 7)
    assert subject.standard_deviation == pytest.approx(
        np.std([-0.1, 0.0, 0.5, 0.7, 0.8, 0.9, 1.1])
    )
    assert subject.zero_fraction == pytest.approx(1 / 7)
    assert subject.negative_fraction == pytest.approx(1 / 7)
    assert subject.above_one_fraction == pytest.approx(1 / 7)
    assert subject.nonfinite_fraction == pytest.approx(1 / 8)

    output = io.StringIO()
    write_csv(statistics, output)
    rows = list(csv.reader(io.StringIO(output.getvalue())))
    assert rows[0][0:3] == ["subject_id", "brain_voxels", "minimum"]
    assert rows[1][0:2] == ["subject-1", "8"]


@pytest.mark.parametrize(
    ("volume", "brain_mask", "variants", "message"),
    [
        (np.ones((2, 2, 2)), np.ones((3, 2, 2)), True, "shapes do not match"),
        (np.ones((2, 2, 2)), np.zeros((2, 2, 2)), True, "no foreground"),
        (
            np.full((2, 2, 2), np.nan),
            np.ones((2, 2, 2)),
            True,
            "no finite",
        ),
        (np.ones((2, 2, 2)), np.ones((2, 2, 2)), False, "no variants"),
    ],
)
def test_inspect_manifest_rejects_invalid_subject_data(
    tmp_path: Path,
    volume: np.ndarray,
    brain_mask: np.ndarray,
    variants: bool,
    message: str,
) -> None:
    manifest_path = write_manifest(
        tmp_path, volume, brain_mask, variants=variants
    )

    with pytest.raises(ApplicationError, match=message):
        inspect_manifest(manifest_path)


def test_main_writes_csv(tmp_path: Path, capsys) -> None:
    manifest_path = write_manifest(
        tmp_path, np.ones((2, 2, 2)), np.ones((2, 2, 2))
    )

    main([str(manifest_path)])

    output = capsys.readouterr().out
    assert output.startswith("subject_id,brain_voxels,minimum,p01")
    assert "subject-1,8,1.0" in output