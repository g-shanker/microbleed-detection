from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.frst.config import (
    PIPELINE_NAME,
    ExperimentConfig,
    expand_variants,
    load_config,
)
from experiments.frst.variants import BASELINE, FrstSpec


def config(**values) -> ExperimentConfig:
    return ExperimentConfig(dataset_dir=Path("dataset"), **values)


def names(**values) -> list[str]:
    return [name for name, _ in expand_variants(config(**values))]


def test_the_example_config_is_valid_and_expands_to_variants() -> None:
    example = Path(__file__).parents[1] / "config.example.toml"

    assert len(expand_variants(load_config(example))) > 1


def test_the_unmodified_pipeline_is_always_the_first_variant() -> None:
    assert expand_variants(config()) == [(PIPELINE_NAME, BASELINE)]


def test_one_at_a_time_sweeps_change_a_single_field_each() -> None:
    variants = dict(
        expand_variants(config(sweep={"alpha": [1, 3], "edge_margin": [2]}))
    )

    assert variants["alpha=1"] == FrstSpec(alpha=1)
    assert variants["alpha=3"] == FrstSpec(alpha=3)
    assert variants["edge_margin=2"] == FrstSpec(edge_margin=2)
    assert len(variants) == 4


def test_grid_sweeps_try_every_combination() -> None:
    variants = dict(
        expand_variants(
            config(sweep_mode="grid", sweep={"alpha": [1, 3], "edge_margin": [1, 2]})
        )
    )

    assert variants["alpha=3,edge_margin=2"] == FrstSpec(alpha=3, edge_margin=2)
    assert len(variants) == 5


def test_sweeps_start_from_the_base_overrides() -> None:
    variants = dict(
        expand_variants(config(base={"edge_margin": 2}, sweep={"alpha": [3]}))
    )

    assert variants["base"] == FrstSpec(edge_margin=2)
    assert variants["alpha=3"] == FrstSpec(edge_margin=2, alpha=3)


def test_custom_variants_may_be_named() -> None:
    assert names(variants=[{"name": "mine", "alpha": 3}, {"alpha": 4}]) == [
        PIPELINE_NAME,
        "mine",
        "custom-2",
    ]


def test_variants_equal_to_an_earlier_one_are_dropped() -> None:
    assert names(sweep={"alpha": [2, 3]}) == [PIPELINE_NAME, "alpha=3"]


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate variant name: same"):
        expand_variants(
            config(
                variants=[{"name": "same", "alpha": 3}, {"name": "same", "alpha": 4}]
            )
        )


def test_invalid_spec_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        expand_variants(config(sweep={"gradient_quantile": [1.5]}))


def test_load_config_reads_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('dataset_dir = "data"\nmax_subjects = 3\n[sweep]\nalpha = [3]\n')

    loaded = load_config(path)

    assert loaded.dataset_dir == Path("data")
    assert loaded.max_subjects == 3
    assert loaded.sweep == {"alpha": [3]}


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('dataset_dir = "data"\nunknown = 1\n')

    with pytest.raises(ValidationError):
        load_config(path)
