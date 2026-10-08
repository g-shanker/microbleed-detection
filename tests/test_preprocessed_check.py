import csv
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
)
from microbleednet.preprocessed_check import inspect_manifest, main, write_csv


def save_volume(path: Path, values: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    return str(path)


def write_manifest(
    tmp_path: Path,
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    include_mask: bool = True,
) -> Path:
    volume_path = save_volume(tmp_path / "volume.nii.gz", volume)
    frst_path = save_volume(tmp_path / "frst.nii.gz", volume)
    mask_path = save_volume(tmp_path / "mask.nii.gz", mask)
    brain_path = save_volume(tmp_path / "brain.nii.gz", np.ones_like(mask))
    manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        subjects=[
            PreprocessedSubject(
                subject_id="subject-1",
                original_volume_path=volume_path,
                brain_mask_path=brain_path,
                bounding_box=((0, 2), (0, 2), (0, 2)),
                variants=[
                    PreprocessedVariant(
                        volume_path=volume_path,
                        mask_path=mask_path if include_mask else None,
                        frst_path=frst_path,
                    )
                ],
            )
        ],
        augmentation_factor=1,
        bias_field_correction="fast",
        raw_manifest_fingerprint="fingerprint",
    )
    path = tmp_path / "preprocessed.json"
    manifest.write(path)
    return path


def test_inspect_manifest_checks_every_artifact(tmp_path: Path) -> None:
    values = np.array(
        [[[-0.1, 0.0], [0.5, 1.1]], [[0.2, 0.3], [0.4, 0.6]]]
    )
    checks = inspect_manifest(
        write_manifest(tmp_path, values, np.zeros((2, 2, 2)))
    )

    assert [check.artifact for check in checks] == [
        "brain_mask",
        "volume",
        "frst",
        "lesion_mask",
    ]
    assert checks[1].negative_count == 1
    assert checks[1].above_one_count == 1
    assert all(not check.issue for check in checks)

    output = io.StringIO()
    write_csv(checks, output)
    rows = list(csv.reader(io.StringIO(output.getvalue())))
    assert rows[0][0:3] == ["subject_id", "variant_index", "artifact"]
    assert rows[1][0:3] == ["subject-1", "", "brain_mask"]


def test_main_accepts_valid_data_without_lesion_mask(
    tmp_path: Path, capsys
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        include_mask=False,
    )

    main([str(manifest_path)])

    captured = capsys.readouterr()
    assert "subject-1,0,volume" in captured.out
    assert "lesion_mask" not in captured.out
    assert not captured.err


def test_main_fails_for_nonfinite_and_nonbinary_data(
    tmp_path: Path, capsys
) -> None:
    values = np.ones((2, 2, 2))
    values[0, 0, 0] = np.nan
    mask = np.zeros((2, 2, 2))
    mask[0, 0, 0] = 2
    manifest_path = write_manifest(tmp_path, values, mask)

    with pytest.raises(SystemExit, match="1"):
        main([str(manifest_path)])

    captured = capsys.readouterr()
    assert "non-finite" in captured.out
    assert "values other than 0 and 1" in captured.out
    assert "Found issues in 3 artifacts" in captured.err


def test_inspect_manifest_reports_shape_and_dimension_issues(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path, np.ones((2, 2, 2, 1)), np.zeros((2, 2, 2))
    )

    checks = inspect_manifest(manifest_path)

    assert "expected 3 dimensions" in checks[1].issue
    assert "expected shape" in checks[1].issue