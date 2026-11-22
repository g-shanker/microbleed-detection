from pathlib import Path

import pytest

from microbleednet.errors import ApplicationError
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSubject,
)
from microbleednet.orchestration.pipes.index_data import (
    build_subject_map,
    execute,
    extract_subject_id,
    find_matching_paths,
    index_source,
    merge_source,
)
from tests.support import create_volume_source, make_index_config, raw_source

VOLUME_PATTERN = "{subject_id}_volume.nii.gz"
REPEATED_PATTERN = "{subject_id}/{subject_id}_volume.nii.gz"


def _subject(subject_id: str, source_id: str) -> RawSubject:
    return RawSubject(
        subject_id=subject_id, source_id=source_id, volume_path=f"/{subject_id}"
    )


def _manifest(source_id: str, *subject_ids: str) -> RawDatasetManifest:
    return RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[raw_source(source_id)],
        subjects=[_subject(subject_id, source_id) for subject_id in subject_ids],
    )


@pytest.mark.parametrize(
    ("relative_path", "pattern", "expected"),
    [
        ("nested/subject_1_volume.nii.gz", "nested/" + VOLUME_PATTERN, "subject_1"),
        ("subject_1_mask.nii.gz", VOLUME_PATTERN, None),
        ("subject_1/subject_1_volume.nii.gz", REPEATED_PATTERN, "subject_1"),
    ],
    ids=["relative-pattern", "nonmatch", "matching-repeated-placeholders"],
)
def test_extract_subject_id(
    tmp_path: Path, relative_path: str, pattern: str, expected: str | None
) -> None:
    assert extract_subject_id(tmp_path, tmp_path / relative_path, pattern) == expected


def test_extract_subject_id_rejects_mismatched_repeated_placeholders(
    tmp_path: Path,
) -> None:
    path = tmp_path / "subject_1" / "subject_2_volume.nii.gz"

    with pytest.raises(ValueError, match="inconsistent subject IDs"):
        extract_subject_id(tmp_path, path, REPEATED_PATTERN)


def test_find_matching_paths_treats_pattern_text_literally(tmp_path: Path) -> None:
    expected = tmp_path / "scan[1]_subject_1.nii.gz"
    expected.write_bytes(b"")
    (tmp_path / "scan1_subject_2.nii.gz").write_bytes(b"")

    assert find_matching_paths(tmp_path, "scan[1]_{subject_id}.nii.gz") == [expected]


@pytest.mark.parametrize(
    ("filenames", "message"),
    [
        (["_volume.nii.gz"], "could not be extracted"),
        (["subject_1_volume.nii.gz"] * 2, "duplicate subject IDs"),
    ],
    ids=["empty-subject-id", "duplicate-subject-id"],
)
def test_build_subject_map_rejects_invalid_subject_ids(
    tmp_path: Path, filenames: list[str], message: str
) -> None:
    paths = [tmp_path / name for name in filenames]

    with pytest.raises(ApplicationError, match=message):
        build_subject_map(tmp_path, paths, VOLUME_PATTERN, "Indexing volumes")


def test_index_source_sorts_naturally_and_namespaces_subjects(tmp_path: Path) -> None:
    source_dir = create_volume_source(tmp_path, "source", ["subject_10", "subject_2"])
    config = make_index_config(
        tmp_path,
        input_dir=source_dir,
        mask_dir=tmp_path / "source_masks",
        source_id="siteA",
        modality="SWI",
    )

    source, subjects = index_source(config)

    assert [subject.subject_id for subject in subjects] == [
        "siteA_subject_2",
        "siteA_subject_10",
    ]
    assert source.source_id == "siteA"
    assert source.modality == "SWI"
    assert {subject.source_id for subject in subjects} == {"siteA"}
    assert all(subject.mask_path for subject in subjects)


def test_index_source_allows_missing_masks(tmp_path: Path) -> None:
    source_dir = create_volume_source(tmp_path, "source", ["subject_1"])
    config = make_index_config(
        tmp_path, input_dir=source_dir, mask_dir=None, mask_pattern=None
    )

    source, subjects = index_source(config)

    assert source.mask_dir is None
    assert source.mask_pattern is None
    assert subjects[0].mask_path is None


@pytest.mark.parametrize(
    ("unmatched_file", "message"),
    [
        ("source_masks/subject_2_mask.nii.gz", "volumes without masks: subject_2"),
        ("source/subject_2_volume.nii.gz", "masks without volumes: subject_2"),
    ],
)
def test_index_source_rejects_unmatched_volumes_and_masks(
    tmp_path: Path, unmatched_file: str, message: str
) -> None:
    source_dir = create_volume_source(tmp_path, "source", ["subject_1", "subject_2"])
    (tmp_path / unmatched_file).unlink()
    config = make_index_config(
        tmp_path, input_dir=source_dir, mask_dir=tmp_path / "source_masks"
    )

    with pytest.raises(ApplicationError, match="IDs do not match") as raised:
        index_source(config)

    assert message in str(raised.value)


def test_merge_source_appends_source_and_sorts_subjects_naturally() -> None:
    existing = _manifest("first", "subject_10")
    addition = _manifest("second", "subject_2")

    merged = merge_source(
        existing, source=addition.sources[0], subjects=addition.subjects
    )

    assert [source.source_id for source in merged.sources] == ["first", "second"]
    assert [subject.subject_id for subject in merged.subjects] == [
        "subject_2",
        "subject_10",
    ]


def test_merge_source_rejects_existing_source_id() -> None:
    existing = _manifest("first", "first_subject_1")
    duplicate = _manifest("first", "first_subject_2")

    with pytest.raises(ValueError) as error:
        merge_source(existing, source=duplicate.sources[0], subjects=duplicate.subjects)

    message = str(error.value)
    assert "Source ID already exists" in message
    assert "new or empty dataset directory" in message
    assert "replacement workflow" not in message


def test_execute_creates_manifest_then_merges_new_sources(tmp_path: Path) -> None:
    manifest_path = DatasetLayout(dataset_dir=tmp_path / "dataset").raw_manifest_path()
    for name in ("a", "b"):
        create_volume_source(tmp_path, name, ["subject_1"])
        execute(
            make_index_config(
                tmp_path,
                input_dir=tmp_path / name,
                mask_dir=tmp_path / f"{name}_masks",
                source_id=name,
            )
        )

    manifest = RawDatasetManifest.read(manifest_path)
    assert [source.source_id for source in manifest.sources] == ["a", "b"]
    assert [subject.subject_id for subject in manifest.subjects] == [
        "a_subject_1",
        "b_subject_1",
    ]
