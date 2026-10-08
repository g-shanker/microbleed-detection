import logging
from typing import cast

from sklearn.model_selection import train_test_split

from ...core import io
from ...progress import progress
from ..configs import SplitConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    SplitManifest,
    content_fingerprint,
)
from ..utils import resolve_path_string

logger = logging.getLogger(__name__)


def subject_has_microbleed(subject: PreprocessedSubject) -> bool:
    """Return whether a subject's original preprocessed mask contains a lesion."""
    mask_path = cast(str, subject.variants[0].mask_path)
    mask = io.nifti_to_numpy(io.load_volume(mask_path))
    return bool(mask.any())


def execute(config: SplitConfig) -> None:
    """Partition preprocessed subjects and persist the reproducible split."""
    manifest_path = DatasetLayout(
        dataset_dir=config.dataset_dir
    ).preprocessed_manifest_path()
    preprocessed_manifest = PreprocessedDatasetManifest.read(manifest_path)
    has_microbleeds = [
        subject_has_microbleed(subject)
        for subject in progress.track(
            preprocessed_manifest.subjects,
            description="Classifying subjects",
        )
    ]
    (
        train_subjects,
        held_out_subjects,
        _,
        held_out_has_microbleeds,
    ) = train_test_split(
        preprocessed_manifest.subjects,
        has_microbleeds,
        train_size=config.train_size,
        random_state=config.seed,
        stratify=has_microbleeds,
    )
    validation_subjects, test_subjects = train_test_split(
        held_out_subjects,
        train_size=config.validation_size
        / (config.validation_size + config.test_size),
        random_state=config.seed,
        stratify=held_out_has_microbleeds,
    )

    split_manifest_path = ExperimentLayout(
        experiment_dir=config.experiment_dir
    ).split_manifest_path()
    SplitManifest(
        status=ManifestStatus.COMPLETE,
        dataset_dir=resolve_path_string(config.dataset_dir),
        preprocessed_manifest_fingerprint=content_fingerprint(preprocessed_manifest),
        seed=config.seed,
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        train_subject_ids=[subject.subject_id for subject in train_subjects],
        validation_subject_ids=[subject.subject_id for subject in validation_subjects],
        test_subject_ids=[subject.subject_id for subject in test_subjects],
    ).write(split_manifest_path)
    logger.info(
        "Split dataset into train=%d validation=%d test=%d subjects; "
        "seed=%s; manifest written to %s",
        len(train_subjects),
        len(validation_subjects),
        len(test_subjects),
        config.seed,
        split_manifest_path,
    )
