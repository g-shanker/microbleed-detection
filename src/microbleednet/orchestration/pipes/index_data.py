import glob
import re
from pathlib import Path
from typing import Optional

from natsort import natsorted

from .. import manifests
from ..configs import SUBJECT_ID_PLACEHOLDER, IndexDataConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
)


def execute(config: IndexDataConfig) -> None:
    now = manifests.timestamp()
    source, subjects = index_source(config, now)

    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    manifest_path = layout.raw_manifest_path()
    existing = (
        RawDatasetManifest.read(manifest_path) if manifest_path.is_file() else None
    )

    raw_manifest = merge_source(
        existing,
        source=source,
        subjects=subjects,
    )
    raw_manifest.write(manifest_path)


def index_source(
    config: IndexDataConfig, now: str
) -> tuple[RawSource, list[RawSubject]]:
    """
    Index one input/mask directory pair into a source and its subjects.
    """
    volume_paths = compute_paths(config.input_dir, config.volume_pattern)
    mask_paths = compute_paths(config.mask_dir, config.mask_pattern)

    volume_subject_map = build_subject_map(
        config.input_dir, volume_paths, config.volume_pattern
    )
    mask_subject_map = build_subject_map(
        config.mask_dir, mask_paths, config.mask_pattern
    )

    volume_ids = set(volume_subject_map)
    mask_ids = set(mask_subject_map)
    unmatched_volumes = sorted(volume_ids - mask_ids)
    unmatched_masks = sorted(mask_ids - volume_ids)
    if unmatched_volumes or unmatched_masks:
        raise ValueError(
            f"unmatched subjects: volumes={unmatched_volumes}, masks={unmatched_masks}"
        )

    def namespaced(subject_id: str) -> str:
        return f"{config.source_id}_{subject_id}"

    subjects = [
        RawSubject(
            subject_id=namespaced(subject_id),
            source_id=config.source_id,
            volume_path=str(volume_subject_map[subject_id].resolve()),
            mask_path=str(mask_subject_map[subject_id].resolve()),
        )
        for subject_id in natsorted(volume_subject_map)
    ]
    source = RawSource(
        input_dir=str(config.input_dir.resolve()),
        mask_dir=str(config.mask_dir.resolve()),
        volume_pattern=config.volume_pattern,
        mask_pattern=config.mask_pattern,
        source_id=config.source_id,
        modality=config.modality,
        added_on=now,
    )
    return source, subjects


def merge_source(
    existing: RawDatasetManifest | None,
    source: RawSource,
    subjects: list[RawSubject],
) -> RawDatasetManifest:
    """Append a freshly indexed source to ``existing`` (or build the first one).

    Subject IDs are globally unique across sources: a new subject that collides
    with one already indexed is a hard error, so accumulation never silently
    drops or overwrites prior data.
    """
    prior_subjects = existing.subjects if existing else []
    collisions = {subject.subject_id for subject in prior_subjects} & {
        subject.subject_id for subject in subjects
    }
    if collisions:
        raise ValueError(
            "subjects already indexed in this dataset: "
            f"{sorted(collisions)}; index them into a fresh dataset directory "
            "or use a different source_id."
        )

    return RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[*(existing.sources if existing else []), source],
        subjects=natsorted(
            [*prior_subjects, *subjects], key=lambda subject: subject.subject_id
        ),
    )


def build_subject_map(
    root_dir: Path,
    paths: list[Path],
    pattern: str,
) -> dict[str, Path]:
    subject_map: dict[str, Path] = {}
    for path in paths:
        subject_id = extract_subject_id(root_dir, path, pattern) or ""
        if not subject_id.strip():
            raise ValueError(f"path has an empty ID: {path}")
        if subject_id in subject_map:
            raise ValueError(f"duplicate subject ID '{subject_id}' in {root_dir}")
        subject_map[subject_id] = path
    return subject_map


def compute_paths(root_dir: Path, pattern: str) -> list[Path]:
    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    glob_pattern = "*".join(glob.escape(part) for part in pattern_parts)
    return list(root_dir.rglob(glob_pattern))


def extract_subject_id(root_dir: Path, path: Path, pattern: str) -> Optional[str]:
    clean_path = path.relative_to(root_dir).as_posix()

    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    escaped_parts = [re.escape(part) for part in pattern_parts]
    regex_pattern = "^" + "(.*?)".join(escaped_parts) + "$"
    match = re.match(regex_pattern, clean_path)
    if match is None:
        return None

    subject_ids = match.groups()
    if len(set(subject_ids)) != 1:
        raise ValueError(
            f"subject ID placeholders do not match in path: {path}"
        )
    return subject_ids[0]
