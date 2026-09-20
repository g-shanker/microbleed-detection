"""Evaluate final binary detections on the held-out training split."""

import logging
from pathlib import Path

from ...core import io as core_io
from ...core.common.metrics import aggregate_metrics, score_masks
from ...progress import progress
from ..configs import (
    EvaluateConfig,
    EvaluateExperimentConfig,
    EvaluateExplicitConfig,
    InferConfig,
    InferExperimentConfig,
    InferExplicitConfig,
)
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    EvaluatedSubject,
    EvaluateManifest,
    InferManifest,
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
    RawSubject,
    SplitManifest,
)
from ..utils import resolve_path_string, resolve_subjects
from . import infer

logger = logging.getLogger(__name__)


def execute(config: EvaluateConfig) -> None:
    """Infer once for held-out subjects, then score masks against references."""
    if config.experiment is not None:
        execute_experiment(config, config.experiment)
    else:
        assert config.explicit is not None
        execute_explicit(config, config.explicit)


def execute_experiment(
    config: EvaluateConfig, mode: EvaluateExperimentConfig
) -> None:
    """Evaluate the held-out subjects from an experiment split."""
    experiment_layout = ExperimentLayout(experiment_dir=mode.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=mode.dataset_dir)
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        dataset_layout.preprocessed_manifest_path()
    )
    split_manifest = SplitManifest.read(experiment_layout.split_manifest_path())
    subjects = resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.test_subject_ids
    )
    infer_config = InferConfig(
        experiment=InferExperimentConfig(
            experiment_dir=mode.experiment_dir,
            subjects=subjects,
        ),
        device=config.device,
    )
    evaluate_subjects(
        config, mode.dataset_dir, experiment_layout, subjects, infer_config
    )


def execute_explicit(
    config: EvaluateConfig, mode: EvaluateExplicitConfig
) -> None:
    """Evaluate every raw subject with explicit checkpoints."""
    experiment_layout = ExperimentLayout(experiment_dir=mode.output_dir)
    dataset_layout = DatasetLayout(dataset_dir=mode.dataset_dir)
    raw_manifest = RawDatasetManifest.read(dataset_layout.raw_manifest_path())
    infer_config = InferConfig(
        explicit=InferExplicitConfig(
            output_dir=mode.output_dir,
            dataset_dir=mode.dataset_dir,
            detector_checkpoint_path=mode.detector_checkpoint_path,
            student_checkpoint_path=mode.student_checkpoint_path,
        ),
        device=config.device,
    )
    evaluate_subjects(
        config,
        mode.dataset_dir,
        experiment_layout,
        raw_manifest.subjects,
        infer_config,
    )


def evaluate_subjects(
    config: EvaluateConfig,
    dataset_dir: Path,
    experiment_layout: ExperimentLayout,
    subjects: list[PreprocessedSubject] | list[RawSubject],
    infer_config: InferConfig,
) -> None:
    infer.execute(infer_config)
    inference_manifest_path = experiment_layout.inference_manifest_path()
    inference_manifest = InferManifest.read(inference_manifest_path)
    if inference_manifest.status is not ManifestStatus.COMPLETE:
        raise ValueError(
            f"manifest at {inference_manifest_path} has status "
            f"{inference_manifest.status.value!r}; a consumer may only read a "
            "complete manifest."
        )
    inference_output_paths = {
        subject.subject_id: subject.output_path
        for subject in inference_manifest.subjects
    }

    per_subject: list[EvaluatedSubject] = []
    for subject in progress.track(subjects, "Evaluating subjects"):
        prediction = core_io.nifti_to_numpy(
            core_io.load_volume(inference_output_paths[subject.subject_id])
        )
        mask_path = (
            subject.mask_path
            if isinstance(subject, RawSubject)
            else subject.variants[0].mask_path
        )
        assert mask_path is not None
        reference = core_io.nifti_to_numpy(core_io.load_volume(mask_path))
        per_subject.append(
            EvaluatedSubject(
                subject_id=subject.subject_id,
                metrics=score_masks(prediction, reference),
            )
        )

    aggregate = aggregate_metrics(subject.metrics for subject in per_subject)
    EvaluateManifest(
        status=ManifestStatus.COMPLETE,
        dataset_dir=resolve_path_string(dataset_dir),
        device=config.device,
        inference_manifest_path=resolve_path_string(
            experiment_layout.inference_manifest_path()
        ),
        subjects=per_subject,
        aggregate=aggregate,
    ).write(experiment_layout.evaluation_manifest_path())
    logger.info(
        "Evaluation complete for %d subjects; manifest written to %s.",
        len(per_subject),
        experiment_layout.evaluation_manifest_path(),
    )