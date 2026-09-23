import glob
import logging
import re
from pathlib import Path
from typing import Optional

from natsort import natsorted

from ...progress import progress
from ..configs import SUBJECT_ID_PLACEHOLDER, IndexDataConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
)
from ..utils import resolve_path_string

logger = logging.getLogger(__name__)


def execute(config: IndexDataConfig) -> None:
    logger.info(
        "Indexing source %s from %s (masks=%s)",
        config.source_id,
        config.input_dir,
        config.mask_dir is not None,
    )
    source, subjects = index_source(config)

    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    manifest_path = layout.raw_manifest_path()

    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = RawDatasetManifest.read(manifest_path)

    if existing_manifest is None:
        raw_manifest = RawDatasetManifest(
            status=ManifestStatus.COMPLETE,
            sources=[source],
            subjects=subjects,
        )
    else:
        raw_manifest = merge_source(
            existing_manifest,
            source=source,
            subjects=subjects,
        )

    raw_manifest.write(manifest_path)
    manifest_action = "created" if existing_manifest is None else "merged"
    logger.info(
        "Indexed %d subjects from source %s; raw manifest %s at %s",
        len(subjects),
        config.source_id,
        manifest_action,
        manifest_path,
    )


def index_source(config: IndexDataConfig) -> tuple[RawSource, list[RawSubject]]:
    """
    Index one input/mask directory pair into a source and its subjects.
    """
    volume_paths = find_matching_paths(config.input_dir, config.volume_pattern)
    volume_subject_map = build_subject_map(
        config.input_dir,
        volume_paths,
        config.volume_pattern,
        description="Indexing volumes",
    )

    mask_subject_map: dict[str, Path] = {}
    if config.mask_dir is not None and config.mask_pattern is not None:
        mask_paths = find_matching_paths(config.mask_dir, config.mask_pattern)
        mask_subject_map = build_subject_map(
            config.mask_dir,
            mask_paths,
            config.mask_pattern,
            description="Indexing masks",
        )

        volume_ids = set(volume_subject_map)
        mask_ids = set(mask_subject_map)
        unmatched_volumes = natsorted(volume_ids - mask_ids)
        unmatched_masks = natsorted(mask_ids - volume_ids)
        if unmatched_volumes or unmatched_masks:
            raise ValueError(
                "unmatched subjects between volumes and masks:\n"
                "  volumes without masks: "
                f"{', '.join(unmatched_volumes) or 'none'}\n"
                "  masks without volumes: "
                f"{', '.join(unmatched_masks) or 'none'}"
            )

    def namespaced(subject_id: str) -> str:
        return f"{config.source_id}_{subject_id}"

    subjects = [
        RawSubject(
            subject_id=namespaced(subject_id),
            source_id=config.source_id,
            volume_path=resolve_path_string(volume_subject_map[subject_id]),
            mask_path=(
                resolve_path_string(mask_subject_map[subject_id])
                if mask_subject_map
                else None
            ),
        )
        for subject_id in natsorted(volume_subject_map)
    ]
    source = RawSource(
        input_dir=resolve_path_string(config.input_dir),
        mask_dir=(
            resolve_path_string(config.mask_dir)
            if config.mask_dir is not None
            else None
        ),
        volume_pattern=config.volume_pattern,
        mask_pattern=config.mask_pattern,
        source_id=config.source_id,
        modality=config.modality,
    )
    return source, subjects


def merge_source(
    existing: RawDatasetManifest,
    source: RawSource,
    subjects: list[RawSubject],
) -> RawDatasetManifest:
    """Append a freshly indexed source to an existing manifest.

    Source IDs are unique across the manifest, which keeps the namespaced
    subject IDs unique as sources are accumulated.
    """
    prior_subjects = existing.subjects
    prior_sources = existing.sources
    if any(
        prior_source.source_id == source.source_id for prior_source in prior_sources
    ):
        raise ValueError(
            f"source already indexed in this dataset: {source.source_id}; "
            "use a different source_id."
        )

    return RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        sources=[*prior_sources, source],
        subjects=natsorted(
            [*prior_subjects, *subjects], key=lambda subject: subject.subject_id
        ),
    )


def build_subject_map(
    root_dir: Path,
    paths: list[Path],
    pattern: str,
    description: str,
) -> dict[str, Path]:
    subject_map: dict[str, Path] = {}
    for path in progress.track(paths, description=description):
        subject_id = extract_subject_id(root_dir, path, pattern) or ""
        if not subject_id.strip():
            raise ValueError(f"path has an empty ID: {path}")
        if subject_id in subject_map:
            raise ValueError(f"duplicate subject ID '{subject_id}' in {root_dir}")
        subject_map[subject_id] = path
    return subject_map


def find_matching_paths(root_dir: Path, pattern: str) -> list[Path]:
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
        raise ValueError(f"subject ID placeholders do not match in path: {path}")
    return subject_ids[0]
