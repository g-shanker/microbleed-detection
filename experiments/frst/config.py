"""Experiment configuration and expansion of the FRST variants to evaluate."""

import itertools
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .variants import FrstSpec

PIPELINE_NAME = "pipeline"


class ExperimentConfig(BaseModel):
    """Settings of one FRST experiment run (see ``config.example.toml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_dir: Path = Field(description="Preprocessed dataset to evaluate on.")
    output_dir: Path = Field(
        default=Path("experiments/frst/results"),
        description="Directory that receives results.csv and variants.json.",
    )
    max_subjects: int | None = Field(
        default=None, gt=0, description="Random subset size; omit to use all subjects."
    )
    seed: int = Field(default=0, description="Seed of the subject subset.")
    lesion_margin: int = Field(
        default=1, ge=0, description="Voxels a lesion is widened by before scoring."
    )
    exclusion_radius: int = Field(
        default=4, ge=0, description="Voxels around lesions with no background peaks."
    )
    peak_min_distance: int = Field(
        default=3, ge=1, description="Minimum voxel distance between background peaks."
    )
    recall_at: tuple[int, ...] = Field(
        default=(5, 20), min_length=1, description="K values of the recall@K metric."
    )
    sweep_mode: Literal["one_at_a_time", "grid"] = Field(
        default="one_at_a_time",
        description=(
            "'one_at_a_time' varies each swept field from the base on its own; "
            "'grid' evaluates every combination of the swept values."
        ),
    )
    base: dict[str, Any] = Field(
        default_factory=dict,
        description="FrstSpec overrides every swept variant starts from.",
    )
    sweep: dict[str, list[Any]] = Field(
        default_factory=dict, description="FrstSpec fields mapped to values to try."
    )
    variants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extra hand-written variants; an optional 'name' key names one.",
    )


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment TOML file."""
    with path.open("rb") as file:
        return ExperimentConfig.model_validate(tomllib.load(file))


def expand_variants(config: ExperimentConfig) -> list[tuple[str, FrstSpec]]:
    """Return the named, de-duplicated variants of ``config``.

    The unmodified pipeline FRST always comes first as the reference.
    """
    candidates: list[tuple[str, dict[str, Any]]] = [(PIPELINE_NAME, {})]
    if config.base:
        candidates.append(("base", dict(config.base)))

    if config.sweep_mode == "grid":
        keys = list(config.sweep)
        for values in itertools.product(*(config.sweep[key] for key in keys)):
            overrides = dict(zip(keys, values))
            candidates.append((_label(overrides), {**config.base, **overrides}))
    else:
        for key, values in config.sweep.items():
            for value in values:
                candidates.append((_label({key: value}), {**config.base, key: value}))

    for index, custom in enumerate(config.variants, start=1):
        overrides = dict(custom)
        name = str(overrides.pop("name", f"custom-{index}"))
        candidates.append((name, {**config.base, **overrides}))

    variants: list[tuple[str, FrstSpec]] = []
    seen: set[FrstSpec] = set()
    names: set[str] = set()
    for name, overrides in candidates:
        spec = FrstSpec.model_validate(overrides)
        if spec in seen:
            continue
        if name in names:
            raise ValueError(f"Duplicate variant name: {name}")
        seen.add(spec)
        names.add(name)
        variants.append((name, spec))
    return variants


def _label(overrides: dict[str, Any]) -> str:
    """Name a swept variant after the values it changes."""
    return ",".join(f"{key}={value}" for key, value in overrides.items())
