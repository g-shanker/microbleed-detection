"""Tests for the CLI command table: registration, wiring, and the generic
``describe`` machinery, plus end-to-end coverage of the ``index-data``
command it currently registers.
"""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import typer
from click.testing import Result
from pydantic import BaseModel, Field
from rich.console import Console
from typer.testing import CliRunner

from microbleednet.cli.commands import (
    build_command,
    build_describe_command,
)
from microbleednet.cli.entrypoint import app, render_application_error
from microbleednet.cli.utils import CommandSpec, is_pipe_module
from microbleednet.errors import ApplicationError
from tests.support import create_volume_source, write_index_config

runner = CliRunner()


class _Nested(BaseModel):
    knob: int = Field(default=1, description="A nested knob.")


class _FakeConfig(BaseModel):
    top_level: str = Field(description="A top-level field.")
    nested: _Nested = Field(default_factory=_Nested)
    trailing: bool = Field(default=True, description="A trailing top-level field.")


def test_describe_renders_a_section_header_for_nested_config() -> None:
    """``build_describe_command`` groups fields by nested-model section.

    Exercised against a scratch Typer app with a synthetic nested config,
    since ``IndexDataConfig`` is flat and never triggers this grouping.
    """
    scratch_app = typer.Typer()

    # A second command keeps Typer from collapsing this scratch app down to a
    # single-command app, which would swallow the "describe" subcommand name.
    @scratch_app.command("noop")
    def noop() -> None:
        pass

    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
    )
    build_describe_command(scratch_app, [spec], Console())

    result = runner.invoke(scratch_app, ["describe", "fake"])
    assert result.exit_code == 0, result.output
    assert "nested" in result.output
    assert "knob" in result.output
    assert "trailing" in result.output


def test_build_command_runs_without_progress_reporter(
    tmp_path: Path, monkeypatch
) -> None:
    scratch_app = typer.Typer()
    executed: list[str] = []
    fake_pipe = SimpleNamespace(
        execute=lambda config: executed.append(config.top_level)
    )
    monkeypatch.setattr(
        "microbleednet.cli.commands.import_module",
        lambda *args, **kwargs: fake_pipe,
    )
    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
    )
    build_command(scratch_app, spec)
    config_path = tmp_path / "config.toml"
    config_path.write_text('top_level = "value"', encoding="utf-8")

    result = runner.invoke(scratch_app, ["--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert executed == ["value"]


def test_build_command_rejects_pipe_without_execute(
    tmp_path: Path, monkeypatch
) -> None:
    scratch_app = typer.Typer()
    monkeypatch.setattr(
        "microbleednet.cli.commands.import_module",
        lambda *args, **kwargs: ModuleType("broken"),
    )
    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
    )
    build_command(scratch_app, spec)
    config_path = tmp_path / "config.toml"
    config_path.write_text('top_level = "value"', encoding="utf-8")

    result = runner.invoke(scratch_app, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert result.exception is not None
    assert "does not expose a callable execute" in str(result.exception)


def test_build_command_renders_application_error(
    tmp_path: Path, monkeypatch
) -> None:
    scratch_app = typer.Typer()
    fake_pipe = SimpleNamespace(
        execute=lambda config: (_ for _ in ()).throw(
            ApplicationError(
                category="Input data",
                summary="Input data is invalid",
                cause="The test fixture is incomplete.",
                fix="Provide the missing input.",
                context={"path": "input.nii.gz"},
            )
        )
    )
    monkeypatch.setattr(
        "microbleednet.cli.commands.import_module",
        lambda *args, **kwargs: fake_pipe,
    )
    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
    )
    build_command(scratch_app, spec)
    config_path = tmp_path / "config.toml"
    config_path.write_text('top_level = "value"', encoding="utf-8")

    result = runner.invoke(scratch_app, ["--config", str(config_path)])

    assert result.exit_code == 1
    assert "Input data is invalid" in result.output
    assert "The test fixture is incomplete." in result.output
    assert "Provide the missing input." in result.output
    assert "input.nii.gz" in result.output

def test_build_command_renders_missing_config_error(tmp_path: Path) -> None:
    scratch_app = typer.Typer()
    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
    )
    build_command(scratch_app, spec)

    result = runner.invoke(
        scratch_app, ["--config", str(tmp_path / "missing.toml")]
    )

    assert result.exit_code == 1
    assert "Configuration file not found" in result.output


def test_render_application_error_supports_summary_only() -> None:
    console = Console(record=True)
    render_application_error(
        ApplicationError(category="Internal", summary="Operation failed"),
        target_console=console,
    )

    output = console.export_text()
    assert "Internal error" in output
    assert "Operation failed" in output


def test_render_application_error_includes_traceback_in_verbose_mode() -> None:
    console = Console(record=True)
    try:
        raise RuntimeError("root failure")
    except RuntimeError as cause:
        error = ApplicationError(
            category="Input data",
            summary="Input data is invalid",
            cause="The test fixture is incomplete.",
            fix="Provide the missing input.",
        )
        error.__cause__ = cause

    render_application_error(error, target_console=console, verbose=True)

    output = console.export_text()
    assert "Details:" in output
    assert "RuntimeError: root failure" in output


def test_is_pipe_module_rejects_module_without_execute() -> None:
    assert not is_pipe_module(ModuleType("broken"))


def test_index_data_help_points_to_describe() -> None:
    result = runner.invoke(app, ["index-data", "--help"])
    assert result.exit_code == 0, result.output


def test_infer_help_points_to_describe() -> None:
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0, result.output
    assert "describe infer" in " ".join(result.output.split())


def test_describe_infer_lists_explicit_checkpoint_keys() -> None:
    result = runner.invoke(app, ["describe", "infer"])
    assert result.exit_code == 0, result.output
    assert "detector_checkpoint_path" in result.output
    assert "student_checkpoint_path" in result.output


def test_describe_index_data_lists_config_keys() -> None:
    result = runner.invoke(app, ["describe", "index-data"])
    assert result.exit_code == 0, result.output
    assert "dataset_dir" in result.output
    assert "volume_pattern" in result.output
    assert "Allowed values: 'T2*-GRE', 'SWI', 'QSM'." in result.output
    assert "complete filename" in result.output
    assert "extension" in result.output


def test_describe_split_lists_partition_keys() -> None:
    result = runner.invoke(app, ["describe", "split"])
    assert result.exit_code == 0, result.output
    assert "train_size" in result.output
    assert "validation_size" in result.output
    assert "test_size" in result.output
    assert "seed" in result.output


def test_describe_rejects_unknown_command() -> None:
    result = runner.invoke(app, ["describe", "bogus"])
    assert result.exit_code != 0
    assert "Unknown command 'bogus'" in result.output


def test_index_data_rejects_missing_mask_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "index.toml"
    write_index_config(
        config_path,
        mask_dir=tmp_path / "missing_masks",
        mask_pattern="{subject_id}_mask.nii.gz",
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Configuration validation failed" in result.output


def test_index_data_rejects_unmatched_subjects_when_masks_required(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "volumes"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    mask_dir.mkdir()
    (input_dir / "subject_1_volume.nii.gz").write_bytes(b"")
    (mask_dir / "subject_2_mask.nii.gz").write_bytes(b"")
    config_path = tmp_path / "index.toml"
    write_index_config(
        config_path,
        input_dir=input_dir,
        mask_dir=mask_dir,
        mask_pattern="{subject_id}_mask.nii.gz",
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Volume and mask subject IDs do not match" in result.output


def test_preprocess_rejects_missing_dataset_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocess.toml"
    config_path.write_text(
        f"dataset_dir = {json.dumps(str(tmp_path / 'missing'))}\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["preprocess", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "Configuration validation failed" in result.output


def _run_index_data(
    tmp_path: Path,
    dataset_dir: Path,
    input_dir: Path,
    source_id: str = "source",
) -> Result:
    config_path = tmp_path / f"index_{input_dir.name}.toml"
    write_index_config(
        config_path,
        dataset_dir=dataset_dir,
        input_dir=input_dir,
        mask_dir=input_dir.with_name(f"{input_dir.name}_masks"),
        source_id=source_id,
    )
    return runner.invoke(app, ["index-data", "--config", str(config_path)])


def _read_raw_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifests" / "raw.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_index_data_accumulates_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1", "subject_2"])
    second = create_volume_source(tmp_path, "second", ["subject_3"])

    assert _run_index_data(tmp_path, dataset_dir, first, "siteA").exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second, "siteB")
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 2
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["siteA_subject_1", "siteA_subject_2", "siteB_subject_3"]
    assert manifest["created_at"] <= manifest["updated_at"]


def test_index_data_rejects_duplicate_source_id(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first).exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second)
    assert result.exit_code != 0
    assert "Source ID already exists" in result.output

    # The failed run must not have clobbered the existing manifest.
    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 1


def test_index_data_source_id_namespaces_subjects(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first, "siteA").exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second, "siteB")
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["siteA_subject_1", "siteB_subject_1"]
    assert [source["source_id"] for source in manifest["sources"]] == [
        "siteA",
        "siteB",
    ]
