"""Evaluate final binary detections on the held-out training split."""

from ...core import io as core_io
from ...core.common.metrics import aggregate_metrics, score_masks
from ..configs import EvaluateConfig, InferConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    EvaluatedSubject,
    EvaluateManifest,
    InferManifest,
    ManifestStatus,
    PreprocessedDatasetManifest,
    SplitManifest,
)
from ..utils import resolve_path_string, resolve_subjects
from . import infer


def execute(config: EvaluateConfig) -> None:
    """Infer once for held-out subjects, then score masks against references."""
    experiment_layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    split_manifest_path = experiment_layout.split_manifest_path()
    split_manifest = SplitManifest.read(split_manifest_path)
    preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
    preprocessed_manifest = PreprocessedDatasetManifest.read(preprocessed_manifest_path)
    subjects = resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.test_subject_ids
    )

    infer.execute(
        InferConfig(
            subjects=subjects,
            experiment_dir=config.experiment_dir,
            device=config.device,
        )
    )
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
    for subject in subjects:
        variant = subject.variants[0]
        prediction = core_io.nifti_to_numpy(
            core_io.load_volume(inference_output_paths[subject.subject_id])
        )
        reference = core_io.nifti_to_numpy(core_io.load_volume(variant.mask_path))
        per_subject.append(
            EvaluatedSubject(
                subject_id=subject.subject_id,
                metrics=score_masks(prediction, reference),
            )
        )

    aggregate = aggregate_metrics(subject.metrics for subject in per_subject)
    EvaluateManifest(
        status=ManifestStatus.COMPLETE,
        dataset_dir=resolve_path_string(config.dataset_dir),
        device=config.device,
        inference_manifest_path=resolve_path_string(
            experiment_layout.inference_manifest_path()
        ),
        subjects=per_subject,
        aggregate=aggregate,
    ).write(experiment_layout.evaluation_manifest_path())