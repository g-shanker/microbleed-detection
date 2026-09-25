import glob
import logging
import re
from pathlib import Path
from typing import Optional

from natsort import natsorted

from ...errors import ApplicationError
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
    """Index a source and merge its subjects into the raw dataset manifest."""
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
        if unmatched_volumes or unmatched_masks:  # pragma: no branch
            details = []
            if unmatched_volumes:  # pragma: no branch
                details.append(
                    f"volumes without masks: {', '.join(unmatched_volumes[:5])}"
                )
            if unmatched_masks:  # pragma: no branch
                details.append(
                    f"masks without volumes: {', '.join(unmatched_masks[:5])}"
                )
            raise ApplicationError(
                category="Input data",
                summary="Volume and mask subject IDs do not match",
                cause="; ".join(details),
                fix=(
                    "Check input directories and the volume_pattern/mask_pattern "
                    "placeholders"
                ),
            )

    def namespaced(subject_id: str) -> str:
        """Prefix a subject ID with its source namespace."""
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
        raise ApplicationError(
            category="Manifest",
            summary="Source ID already exists",
            cause=f"The manifest already contains source '{source.source_id}'",
            fix=(
                "Choose a new source ID or rebuild the dataset in a new or empty "
                "dataset directory"
            ),
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
    """Map unique subject IDs extracted from matching paths to those paths."""
    subject_map: dict[str, Path] = {}
    for path in progress.track(paths, description=description):
        subject_id = extract_subject_id(root_dir, path, pattern) or ""
        if not subject_id.strip():
            raise ApplicationError(
                category="Input data",
                summary="Subject ID could not be extracted from a path",
                cause=f"Pattern '{pattern}' produced an empty ID for '{path}'",
                fix="Correct the filename pattern so it captures a stable subject ID",
                context={"path": str(path)},
            )
        if subject_id in subject_map:
            raise ApplicationError(
                category="Input data",
                summary="Filename pattern produces duplicate subject IDs",
                cause=f"Subject ID '{subject_id}' was extracted more than once",
                fix="Correct the pattern so each subject ID maps to one file",
                context={"subject_id": subject_id},
            )
        subject_map[subject_id] = path
    return subject_map


def find_matching_paths(root_dir: Path, pattern: str) -> list[Path]:
    """Find paths matching a subject-ID filename pattern beneath a root."""
    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    glob_pattern = "*".join(glob.escape(part) for part in pattern_parts)
    return list(root_dir.rglob(glob_pattern))


def extract_subject_id(root_dir: Path, path: Path, pattern: str) -> Optional[str]:
    """Extract the consistent subject ID captured by a path pattern."""
    clean_path = path.relative_to(root_dir).as_posix()

    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    escaped_parts = [re.escape(part) for part in pattern_parts]
    regex_pattern = "^" + "(.*?)".join(escaped_parts) + "$"
    match = re.match(regex_pattern, clean_path)
    if match is None:
        return None

    subject_ids = match.groups()
    if len(set(subject_ids)) != 1:
        raise ApplicationError(
            category="Input data",
            summary="Filename pattern captures inconsistent subject IDs",
            cause=f"Pattern '{pattern}' captures different IDs in '{path}'",
            fix="Use one consistent {subject_id} capture in the pattern",
            context={"path": str(path)},
        )
    return subject_ids[0]
