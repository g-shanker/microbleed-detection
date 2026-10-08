import csv
import io
from pathlib import Path

import numpy as np
import pytest

from microbleednet.core.datamodels import PatchRecord
from microbleednet.orchestration.manifests import ManifestStatus, PatchManifest
from microbleednet.patch_check import inspect_manifest, main, write_csv


def write_manifest(
    tmp_path: Path,
    volumes: np.ndarray,
    masks: np.ndarray,
    frst: np.ndarray,
    records: list[PatchRecord] | None = None,
) -> Path:
    volume_path = tmp_path / "volume.npy"
    mask_path = tmp_path / "mask.npy"
    frst_path = tmp_path / "frst.npy"
    np.save(volume_path, volumes)
    np.save(mask_path, masks)
    np.save(frst_path, frst)
    if records is None:
        records = [
            PatchRecord(
                volume_path=str(volume_path),
                mask_path=str(mask_path),
                frst_path=str(frst_path),
                patch_index=index,
                has_microbleed=bool(np.any(masks[index] > 0)),
            )
            for index in range(len(volumes))
        ]
    manifest = PatchManifest(
        status=ManifestStatus.COMPLETE,
        stage="detector",
        split="train",
        subject_ids=["subject-1"],
        patch_size=2,
        augmentation_factor=1,
        probability_threshold=None,
        use_amp=False,
        records=records,
    )
    path = tmp_path / "patches.json"
    manifest.write(path)
    return path


def test_inspect_manifest_reports_valid_patch_arrays(tmp_path: Path) -> None:
    volumes = np.arange(16, dtype=float).reshape((2, 2, 2, 2))
    masks = np.zeros_like(volumes)
    masks[1, 0, 0, 0] = 1
    checks = inspect_manifest(write_manifest(tmp_path, volumes, masks, volumes))

    assert len(checks) == 1
    assert checks[0].record_count == 2
    assert checks[0].array_count == 2
    assert checks[0].volume_minimum == 0
    assert checks[0].volume_maximum == 15
    assert not checks[0].issue

    output = io.StringIO()
    write_csv(checks, output)
    rows = list(csv.reader(io.StringIO(output.getvalue())))
    assert rows[0][0:3] == ["volume_path", "mask_path", "frst_path"]
    assert rows[1][3:6] == ["2", "2", "2x2x2x2"]


def test_main_accepts_valid_patch_data(tmp_path: Path, capsys) -> None:
    values = np.ones((1, 2, 2, 2))
    manifest_path = write_manifest(tmp_path, values, np.zeros_like(values), values)

    main([str(manifest_path)])

    captured = capsys.readouterr()
    assert ",1,1,1x2x2x2," in captured.out
    assert not captured.err


def test_inspect_manifest_reports_mismatched_label(tmp_path: Path) -> None:
    values = np.ones((1, 2, 2, 2))
    masks = np.zeros_like(values)
    volume_path = tmp_path / "volume.npy"
    mask_path = tmp_path / "mask.npy"
    frst_path = tmp_path / "frst.npy"
    records = [
        PatchRecord(
            volume_path=str(volume_path),
            mask_path=str(mask_path),
            frst_path=str(frst_path),
            patch_index=0,
            has_microbleed=True,
        )
    ]

    issue = inspect_manifest(
        write_manifest(tmp_path, values, masks, values, records)
    )[0].issue

    assert "1 has_microbleed labels do not match masks" in issue


def test_main_fails_for_corrupt_patch_data(tmp_path: Path, capsys) -> None:
    volumes = np.ones((2, 2, 2, 2))
    volumes[0, 0, 0, 0] = np.nan
    masks = np.zeros_like(volumes)
    masks[0, 0, 0, 0] = 2
    frst = np.ones((2, 3, 2, 2))
    manifest_path = write_manifest(tmp_path, volumes, masks, frst)

    with pytest.raises(SystemExit, match="1"):
        main([str(manifest_path)])

    captured = capsys.readouterr()
    assert "volume contains 1 non-finite values" in captured.out
    assert "mask contains values other than 0 and 1" in captured.out
    assert "array shapes differ" in captured.out
    assert "Found issues in 1 patch array groups" in captured.err


def test_inspect_manifest_reports_record_and_dimension_issues(tmp_path: Path) -> None:
    volume_path = tmp_path / "volume.npy"
    mask_path = tmp_path / "mask.npy"
    frst_path = tmp_path / "frst.npy"
    records = [
        PatchRecord(
            volume_path=str(volume_path),
            mask_path=str(mask_path),
            frst_path=str(frst_path),
            patch_index=0,
            has_microbleed=True,
        ),
        PatchRecord(
            volume_path=str(volume_path),
            mask_path=str(mask_path),
            frst_path=str(frst_path),
            patch_index=0,
            has_microbleed=False,
        ),
        PatchRecord(
            volume_path=str(volume_path),
            mask_path=str(mask_path),
            frst_path=str(frst_path),
            patch_index=3,
            has_microbleed=False,
        ),
    ]
    manifest_path = write_manifest(
        tmp_path,
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        np.full((2, 2, 2), np.nan),
        records,
    )

    issue = inspect_manifest(manifest_path)[0].issue

    assert "dimensions instead of 4" in issue
    assert "duplicate patch indices" in issue
    assert "out-of-range index" in issue
    assert "3 records for 2 stored patches" in issue