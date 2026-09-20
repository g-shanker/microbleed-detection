"""Create and persist the subject split consumed by training."""

from sklearn.model_selection import train_test_split

from ..configs import SplitConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    SplitManifest,
    content_fingerprint,
)
from ..utils import resolve_path_string


def execute(config: SplitConfig) -> None:
    manifest_path = DatasetLayout(
        dataset_dir=config.dataset_dir
    ).preprocessed_manifest_path()
    preprocessed_manifest = PreprocessedDatasetManifest.read(manifest_path)
    train_subjects, held_out_subjects = train_test_split(
        preprocessed_manifest.subjects,
        train_size=config.train_size,
        random_state=config.seed,
    )
    validation_subjects, test_subjects = train_test_split(
        held_out_subjects,
        train_size=config.validation_size
        / (config.validation_size + config.test_size),
        random_state=config.seed,
    )

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
    ).write(ExperimentLayout(experiment_dir=config.experiment_dir).split_manifest_path())