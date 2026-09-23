"""Evaluate final binary detections on the held-out training split."""

import logging
from pathlib import Path

from ...core import io as core_io
from ...core.common.metrics import aggregate_metrics, score_masks
from ...progress import progress
from ..configs import EvaluateConfig, InferConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    EvaluatedSubject,
    EvaluateManifest,
    InferManifest,
    ManifestStatus,
    PreprocessedDatasetManifest,
    RawDatasetManifest,
    SplitManifest,
)
from ..utils import resolve_path_string, resolve_subjects
from . import infer

logger = logging.getLogger(__name__)


def execute(config: EvaluateConfig) -> None:
    """Infer once for held-out subjects, then score masks against references."""
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    experiment_layout = ExperimentLayout(experiment_dir=config.output_dir)

    if config.experiment_dir is not None:
        preprocessed_manifest = PreprocessedDatasetManifest.read(
            dataset_layout.preprocessed_manifest_path()
        )
        split_manifest = SplitManifest.read(experiment_layout.split_manifest_path())
        inference_subjects = resolve_subjects(
            preprocessed_manifest.subjects, split_manifest.test_subject_ids
        )
        evaluation_subjects = [
            (subject.subject_id, subject.variants[0].mask_path)
            for subject in inference_subjects
        ]
        infer.infer_subjects(
            config.device,
            experiment_layout,
            config.detector_checkpoint_path,
            config.student_checkpoint_path,
            inference_subjects,
        )
    else:
        raw_manifest = RawDatasetManifest.read(
            dataset_layout.raw_manifest_path()
        )
        evaluation_subjects = [
            (subject.subject_id, subject.mask_path)
            for subject in raw_manifest.subjects
        ]
        infer.execute(
            InferConfig(
                output_dir=config.output_dir,
                dataset_dir=config.dataset_dir,
                detector_checkpoint_path=config.detector_checkpoint_path,
                student_checkpoint_path=config.student_checkpoint_path,
                device=config.device,
            )
        )

    evaluate_subjects(
        config,
        config.dataset_dir,
        experiment_layout,
        evaluation_subjects,
    )


def evaluate_subjects(
    config: EvaluateConfig,
    dataset_dir: Path,
    experiment_layout: ExperimentLayout,
    subjects: list[tuple[str, str | None]],
) -> None:
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
    for subject_id, mask_path in progress.track(subjects, "Evaluating subjects"):
        prediction = core_io.nifti_to_numpy(
            core_io.load_volume(inference_output_paths[subject_id])
        )
        reference = core_io.nifti_to_numpy(core_io.load_volume(mask_path))
        per_subject.append(
            EvaluatedSubject(
                subject_id=subject_id,
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