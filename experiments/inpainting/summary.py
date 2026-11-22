"""Pool per-subject measurements into one row per iteration count."""

import math
from collections.abc import Sequence

import numpy as np

from experiments.frst.scoring import summarize

from .measure import Measurement


def summarize_iterations(
    measurements: Sequence[Measurement], recall_at: tuple[int, ...]
) -> dict[str, float]:
    """Pool one iteration count's measurements over subjects.

    * ``lesion_loss``: share of annotated lesion voxels inside the dilated
      vessel mask, i.e. erased by inpainting. Lower is better.
    * ``lesions_hit``: share of lesions with at least one voxel erased.
    * ``vessel_fraction``: mean share of brain voxels that get inpainted.
    * ``residual_vessels``: the strategy's own vessel detector is run again on
      the inpainted volume; this is the vessel area it still finds relative to
      the area found before inpainting. Lower means fewer vessels survive.
    * ``mean_auc`` and ``recall@K``: FRST-only lesion separation after
      inpainting, as in the FRST experiment. Higher is better.
    """
    lesion_voxels = sum(item.lesion_voxels for item in measurements)
    lesions = sum(item.lesions for item in measurements)
    summary = {
        "lesion_loss": _ratio(
            sum(item.lesion_voxels_removed for item in measurements), lesion_voxels
        ),
        "lesions_hit": _ratio(sum(item.lesions_hit for item in measurements), lesions),
        "vessel_fraction": float(np.mean([m.vessel_fraction for m in measurements])),
        "residual_vessels": _mean_defined([m.residual_vessels for m in measurements]),
    }
    summary.update(summarize([item.scores for item in measurements], recall_at))
    return summary


def recommend(
    rows: Sequence[dict[str, float]], max_lesion_loss: float, auc_tolerance: float
) -> int | None:
    """Return the smallest iteration count that is nearly as good as the best.

    Only counts that erase at most ``max_lesion_loss`` of the lesion voxels are
    considered. Among them, the best ``mean_auc`` sets the bar, and the smallest
    count within ``auc_tolerance`` of it wins. Returns ``None`` if no count
    qualifies.
    """
    allowed = [
        row
        for row in rows
        if row["lesion_loss"] <= max_lesion_loss and not math.isnan(row["mean_auc"])
    ]
    if not allowed:
        return None
    best = max(row["mean_auc"] for row in allowed)
    return int(
        min(
            row["iterations"]
            for row in allowed
            if row["mean_auc"] >= best - auc_tolerance
        )
    )


def _mean_defined(values: Sequence[float]) -> float:
    """Average the defined values; NaN when none are."""
    defined = [value for value in values if not math.isnan(value)]
    return float(np.mean(defined)) if defined else float("nan")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")
