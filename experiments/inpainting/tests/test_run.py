import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from experiments.inpainting import run
from experiments.inpainting.config import InpaintingConfig, load_config
from experiments.inpainting.tests.support import (
    AFFINE,
    LESION,
    SHAPE,
    THROUGH_PLANE,
    VESSEL,
    read_subjects,
    write_config,
    write_raw_dataset,
)
from microbleednet.core.io import load_volume


def read_results(tmp_path: Path) -> list[dict[str, str]]:
    with (tmp_path / "results" / "results.csv").open() as file:
        return list(csv.DictReader(file))


def test_the_example_config_is_valid() -> None:
    example = Path(__file__).parents[1] / "config.example.toml"

    config = load_config(example)

    assert config.iterations == (0, 1, 2, 3, 4, 5, 6)
    assert config.strategies == ("axial", "frangi3d", "frangi3d_range")
    assert config.save_masks and not config.save_inpainted


def test_load_config_reads_toml_and_sorts_iterations(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'dataset_dir = "data"\nbias_field_correction = "n4"\niterations = [3, 1]\n'
    )

    config = load_config(path)

    assert config.iterations == (1, 3)
    assert config.bias_field_correction == "n4"
    assert config.strategies == ("axial", "frangi3d", "frangi3d_range")


@pytest.mark.parametrize(
    "overrides",
    [
        {"iterations": ()},
        {"iterations": (-1, 1)},
        {"iterations": (1, 1)},
        {"strategies": ()},
        {"strategies": ("axial", "axial")},
        {"strategies": ("coronal",)},
        {"save_masks": False, "save_inpainted": True},
    ],
)
def test_invalid_settings_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        InpaintingConfig(
            dataset_dir=Path("data"), bias_field_correction="fast", **overrides
        )


def test_the_bias_field_method_is_required() -> None:
    with pytest.raises(ValidationError, match="bias_field_correction"):
        InpaintingConfig.model_validate({"dataset_dir": "data"})


def test_select_subjects_skips_unmasked_subjects(tmp_path: Path) -> None:
    subjects = read_subjects(write_raw_dataset(tmp_path, ["b", "a"]))
    unmasked = subjects[0].model_copy(update={"subject_id": "c", "mask_path": None})

    selected = run.select_subjects([*subjects, unmasked], None, seed=0)

    assert [subject.subject_id for subject in selected] == ["a", "b"]


def test_select_subjects_takes_a_seeded_subset(tmp_path: Path) -> None:
    subjects = read_subjects(write_raw_dataset(tmp_path, ["a", "b", "c"]))

    first = run.select_subjects(subjects, 2, seed=1)

    assert len(first) == 2
    assert first == run.select_subjects(subjects, 2, seed=1)


def test_main_writes_one_row_per_strategy_and_iteration_count(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a", "b"])

    assert run.main(["--config", str(write_config(tmp_path, dataset))]) == 0

    rows = read_results(tmp_path)
    assert [(row["strategy"], row["iterations"]) for row in rows] == [
        ("axial", "0.0"),
        ("axial", "2.0"),
        ("frangi3d", "0.0"),
        ("frangi3d", "2.0"),
    ]
    assert all(row["subjects"] == "2.0" for row in rows)
    for strategy in ("axial", "frangi3d"):
        marked = [r for r in rows if r["strategy"] == strategy]
        assert [r["recommended"] for r in marked].count("True") == 1
    output = capsys.readouterr().out
    assert "Strategy: axial" in output and "Strategy: frangi3d" in output
    assert "axial: recommended in-plane dilation iterations" in output
    assert len(fake_pipeline) == 2


def test_strategies_can_be_limited(tmp_path: Path, fake_pipeline) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    config = write_config(tmp_path, dataset, strategies=["frangi3d"])

    run.main(["--config", str(config)])

    assert {row["strategy"] for row in read_results(tmp_path)} == {"frangi3d"}


def test_the_range_strategy_is_measured_and_saved_like_the_others(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    config = write_config(
        tmp_path, dataset, strategies=["frangi3d_range"], iterations=[0]
    )

    run.main(["--config", str(config)])

    assert {row["strategy"] for row in read_results(tmp_path)} == {"frangi3d_range"}
    assert (tmp_path / "results" / "masks" / "a" / "frangi3d_range_n0.nii.gz").is_file()


def test_the_strategies_measure_different_masks(tmp_path: Path, fake_pipeline) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset, iterations=[0]))])

    rows = {row["strategy"]: row for row in read_results(tmp_path)}
    assert float(rows["frangi3d"]["vessel_fraction"]) == pytest.approx(
        2 * float(rows["axial"]["vessel_fraction"])
    )


def test_masks_are_saved_for_every_subject_strategy_and_count(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a", "b"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    for subject in ("a", "b"):
        directory = tmp_path / "results" / "masks" / subject
        saved = {path.name for path in directory.iterdir()}
        assert saved == {
            "volume.nii.gz",
            "lesions.nii.gz",
            "brain.nii.gz",
            "axial_n0.nii.gz",
            "axial_n2.nii.gz",
            "frangi3d_n0.nii.gz",
            "frangi3d_n2.nii.gz",
        }


def test_saved_masks_show_what_was_inpainted(tmp_path: Path, fake_pipeline) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset))])

    directory = tmp_path / "results" / "masks" / "a"
    axial = load_volume(directory / "axial_n0.nii.gz")
    three_d = np.asarray(load_volume(directory / "frangi3d_n0.nii.gz").dataobj)
    np.testing.assert_array_equal(axial.affine, AFFINE)
    assert axial.shape == SHAPE
    assert np.asarray(axial.dataobj)[VESSEL] == 1
    assert not np.asarray(axial.dataobj)[THROUGH_PLANE]
    assert three_d[THROUGH_PLANE] == 1
    lesions = np.asarray(load_volume(directory / "lesions.nii.gz").dataobj)
    assert lesions[LESION] == 1


def test_inpainted_volumes_are_only_saved_on_request(
    tmp_path: Path, fake_pipeline
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    config = write_config(
        tmp_path, dataset, iterations=[0], strategies=["axial"], save_inpainted=True
    )

    run.main(["--config", str(config)])

    directory = tmp_path / "results" / "masks" / "a"
    assert (directory / "axial_n0_inpainted.nii.gz").is_file()


def test_nothing_is_saved_when_masks_are_turned_off(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    run.main(["--config", str(write_config(tmp_path, dataset, save_masks=False))])

    assert not (tmp_path / "results" / "masks").exists()
    assert "Wrote masks" not in capsys.readouterr().out


def test_a_second_run_reuses_the_cached_preprocessing(
    tmp_path: Path, fake_pipeline
) -> None:
    config = write_config(tmp_path, write_raw_dataset(tmp_path, ["a"]))

    run.main(["--config", str(config)])
    run.main(["--config", str(config)])

    assert len(fake_pipeline) == 1


def test_the_table_explains_when_nothing_is_recommended(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])
    config = write_config(
        tmp_path, dataset, iterations=[2], strategies=["axial"], max_lesion_loss=0.0
    )

    assert run.main(["--config", str(config)]) == 0

    assert "axial: no iteration count keeps lesion loss" in capsys.readouterr().out


def test_dry_run_lists_the_plan_without_computing(
    tmp_path: Path, fake_pipeline, capsys
) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"])

    config = write_config(tmp_path, dataset)
    assert run.main(["--config", str(config), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "strategies: ['axial', 'frangi3d']" in output
    assert "iterations: [0, 2]" in output
    assert not fake_pipeline
    assert not (tmp_path / "results").exists()


def test_main_fails_when_no_subject_has_a_mask(tmp_path: Path) -> None:
    dataset = write_raw_dataset(tmp_path, ["a"], with_masks=False)

    assert run.main(["--config", str(write_config(tmp_path, dataset))]) == 1
