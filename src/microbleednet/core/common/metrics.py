from typing import Iterable

import numpy as np

from ...constants import COMPONENT_CONNECTIVITY
from ..datamodels import EvaluationAggregate, EvaluationMetrics
from ..utils import label_components


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return foreground voxel Dice, scoring two empty masks as one."""
    prediction_foreground = prediction > 0
    reference_foreground = reference > 0
    foreground_count = int(
        np.count_nonzero(prediction_foreground)
        + np.count_nonzero(reference_foreground)
    )
    intersection = int(
        np.count_nonzero(prediction_foreground & reference_foreground)
    )
    return 2 * intersection / foreground_count if foreground_count else 1.0


def score_masks(
    prediction: np.ndarray, reference: np.ndarray
) -> EvaluationMetrics:
    """Count any-overlap clusters in prevalidated masks on the same voxel grid."""
    prediction_labels = label_components(prediction > 0, COMPONENT_CONNECTIVITY)
    reference_labels = label_components(reference > 0, COMPONENT_CONNECTIVITY)
    overlap = (prediction_labels > 0) & (reference_labels > 0)
    detected_reference_clusters = int(np.unique(reference_labels[overlap]).size)
    overlapping_prediction_clusters = int(np.unique(prediction_labels[overlap]).size)
    reference_count = int(reference_labels.max())
    prediction_count = int(prediction_labels.max())
    return EvaluationMetrics(
        detected_reference_clusters=detected_reference_clusters,
        overlapping_prediction_clusters=overlapping_prediction_clusters,
        false_positive=prediction_count - overlapping_prediction_clusters,
        false_negative=reference_count - detected_reference_clusters,
        cluster_tpr=(
            detected_reference_clusters / reference_count if reference_count else 0.0
        ),
        cluster_precision=(
            overlapping_prediction_clusters / prediction_count
            if prediction_count else 0.0
        ),
        dice_score=dice_score(prediction_labels, reference_labels),
    )


def aggregate_metrics(metrics: Iterable[EvaluationMetrics]) -> EvaluationAggregate:
    """Pool lesion counts and ratios and average subject voxel Dice."""
    subject_metrics = list(metrics)
    detected_reference_clusters = sum(
        metric.detected_reference_clusters for metric in subject_metrics
    )
    overlapping_prediction_clusters = sum(
        metric.overlapping_prediction_clusters for metric in subject_metrics
    )
    false_positive = sum(metric.false_positive for metric in subject_metrics)
    false_negative = sum(metric.false_negative for metric in subject_metrics)
    reference_count = detected_reference_clusters + false_negative
    prediction_count = overlapping_prediction_clusters + false_positive
    subject_count = len(subject_metrics)
    return EvaluationAggregate(
        detected_reference_clusters=detected_reference_clusters,
        overlapping_prediction_clusters=overlapping_prediction_clusters,
        false_positive=false_positive,
        false_negative=false_negative,
        cluster_tpr=(
            detected_reference_clusters / reference_count if reference_count else 0.0
        ),
        cluster_precision=(
            overlapping_prediction_clusters / prediction_count
            if prediction_count else 0.0
        ),
        false_positives_per_subject=(
            false_positive / subject_count if subject_count else 0.0
        ),
        dice_score=(
            sum(metric.dice_score for metric in subject_metrics) / subject_count
            if subject_count else 0.0
        ),
    )
