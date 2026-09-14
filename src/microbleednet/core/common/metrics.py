"""Connected-component metrics for binary detection masks."""

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..datamodels import EvaluationAggregate, EvaluationMetrics
from ..utils import COMPONENT_CONNECTIVITY, label_components


def score_masks(
    prediction: np.ndarray, reference: np.ndarray
) -> EvaluationMetrics:
    """Score connected components with one-to-one overlap matching."""
    prediction_labels = label_components(prediction, COMPONENT_CONNECTIVITY)
    reference_labels = label_components(reference, COMPONENT_CONNECTIVITY)
    prediction_ids = np.unique(prediction_labels[prediction_labels > 0])
    reference_ids = np.unique(reference_labels[reference_labels > 0])
    overlap = np.zeros((len(prediction_ids), len(reference_ids)), dtype=np.int64)
    for prediction_index, prediction_id in enumerate(prediction_ids):
        prediction_region = prediction_labels == prediction_id
        reference_values, counts = np.unique(
            reference_labels[prediction_region], return_counts=True
        )
        for reference_id, count in zip(reference_values, counts):
            if reference_id:
                overlap[prediction_index, reference_ids == reference_id] = count
    if overlap.size:
        rows, columns = linear_sum_assignment(-overlap)
        matched = overlap[rows, columns] > 0
        true_positive = int(np.count_nonzero(matched))
    else:
        true_positive = 0
    false_positive = len(prediction_ids) - true_positive
    false_negative = len(reference_ids) - true_positive
    reference_count = len(reference_ids)
    prediction_count = len(prediction_ids)
    return EvaluationMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        cluster_tpr=(
            true_positive / reference_count if reference_count else 0.0
        ),
        cluster_precision=(
            true_positive / prediction_count if prediction_count else 0.0
        ),
    )


def aggregate_metrics(metrics: Iterable[EvaluationMetrics]) -> EvaluationAggregate:
    """Aggregate lesion counts and ratios across evaluated subjects."""
    subject_metrics = list(metrics)
    true_positive = sum(metric.true_positive for metric in subject_metrics)
    false_positive = sum(metric.false_positive for metric in subject_metrics)
    false_negative = sum(metric.false_negative for metric in subject_metrics)
    reference_count = true_positive + false_negative
    prediction_count = true_positive + false_positive
    subject_count = len(subject_metrics)
    return EvaluationAggregate(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        cluster_tpr=(true_positive / reference_count if reference_count else 0.0),
        cluster_precision=(
            true_positive / prediction_count if prediction_count else 0.0
        ),
        false_positives_per_subject=(
            false_positive / subject_count if subject_count else 0.0
        ),
    )
