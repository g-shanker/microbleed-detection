"""Tests for the CLI command table: registration, wiring, ``describe``, and
end-to-end coverage of the ``index-data`` command.
"""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import typer
import typer.main
from click.testing import Result
from rich.console import Console
from typer.testing import CliRunner

from microbleednet.cli.commands import build_command, build_describe_command
from microbleednet.cli.entrypoint import app
from microbleednet.errors import ApplicationError
from tests.support import create_volume_source, write_index_config
from tests.support_cli_core import (  # noqa: F401
    FAKE_SPEC,
    FakeConfig,
    cli_globals_restored,
)

pytestmark = pytest.mark.usefixtures("cli_globals_restored")

runner = CliRunner()


def raise_application_error(config: FakeConfig) -> None:
    raise ApplicationError(
        category="Input data",
        summary="Input data is invalid",
        cause="The test fixture is incomplete.",
        fix="Provide the missing input.",
        context={"path": "input.nii.gz"},
    )


@pytest.fixture
def fake_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text('top_level = "value"', encoding="utf-8")
    return path


def invoke_fake_command(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, pipe: Any = None
) -> Result:
    """Run a scratch app whose single command imports ``pipe`` instead of a real one."""
    scratch_app = typer.Typer()
    monkeypatch.setattr(
        "microbleednet.cli.commands.import_module", lambda *args, **kwargs: pipe
    )
    build_command(scratch_app, FAKE_SPEC)
    return runner.invoke(scratch_app, ["--config", str(config_path)])


def test_command_executes_pipe_with_parsed_config(
    monkeypatch: pytest.MonkeyPatch, fake_config_path: Path
) -> None:
    executed: list[FakeConfig] = []
    pipe = SimpleNamespace(execute=executed.append)

    result = invoke_fake_command(monkeypatch, fake_config_path, pipe)

    assert result.exit_code == 0, result.output
    assert executed == [FakeConfig(top_level="value")]


def test_command_rejects_pipe_without_execute(
    monkeypatch: pytest.MonkeyPatch, fake_config_path: Path
) -> None:
    result = invoke_fake_command(monkeypatch, fake_config_path, ModuleType("broken"))

    assert result.exit_code != 0
    assert "does not expose a callable execute" in str(result.exception)


def test_command_renders_application_error_from_pipe(
    monkeypatch: pytest.MonkeyPatch, fake_config_path: Path
) -> None:
    pipe = SimpleNamespace(execute=raise_application_error)

    result = invoke_fake_command(monkeypatch, fake_config_path, pipe)

    assert result.exit_code == 1
    for text in (
        "Input data is invalid",
        "The test fixture is incomplete.",
        "Provide the missing input.",
        "input.nii.gz",
    ):
        assert text in result.output


def test_command_renders_missing_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = invoke_fake_command(monkeypatch, tmp_path / "missing.toml")

    assert result.exit_code == 1
    assert "Configuration file not found" in result.output


@pytest.mark.parametrize(
    ("command", "summary"),
    [
        ("index-data", "Index one raw NIfTI source"),
        ("preprocess", "Preprocess every indexed subject"),
        ("split", "Assign preprocessed subjects"),
        ("train", "Train detector, teacher, and student stages"),
        ("infer", "Apply explicit detector and student checkpoints"),
        ("evaluate", "Run inference and lesion-level scoring"),
    ],
)
def test_command_help_summarizes_operation_and_points_to_describe(
    command: str, summary: str
) -> None:
    registered = typer.main.get_command(app).commands[command]  # type: ignore[attr-defined]

    assert summary in " ".join((registered.help or "").split())
    assert f"describe {command}" in (registered.epilog or "")


def test_describe_renders_section_header_for_nested_config() -> None:
    scratch_app = typer.Typer()

    # A second command keeps Typer from collapsing the scratch app to a
    # single-command app, which would swallow the "describe" subcommand name.
    @scratch_app.command("noop")
    def noop() -> None:
        pass

    build_describe_command(scratch_app, [FAKE_SPEC], Console())

    result = runner.invoke(scratch_app, ["describe", "fake"])

    assert result.exit_code == 0, result.output
    for text in ("top_level", "required", "[nested]", "knob", "trailing"):
        assert text in result.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("infer", ["detector_checkpoint_path", "student_checkpoint_path"]),
        (
            "index-data",
            [
                "dataset_dir",
                "volume_pattern",
                "Allowed values: 'T2*-GRE', 'SWI', 'QSM'.",
                "complete filename",
                "extension",
            ],
        ),
        ("split", ["train_size", "validation_size", "test_size", "seed"]),
    ],
)
def test_describe_lists_config_keys(command: str, expected: list[str]) -> None:
    result = runner.invoke(app, ["describe", command])

    assert result.exit_code == 0, result.output
    for text in expected:
        assert text in result.output


def test_describe_rejects_unknown_command() -> None:
    result = runner.invoke(app, ["describe", "bogus"])

    assert result.exit_code != 0
    assert "Unknown command 'bogus'" in result.output


def run_index_data(
    tmp_path: Path, input_dir: Path, source_id: str = "source", **overrides: object
) -> Result:
    """Run ``index-data`` for ``input_dir`` into ``tmp_path / 'dataset'``."""
    config_path = tmp_path / f"index_{input_dir.name}.toml"
    values = {
        "input_dir": input_dir,
        "mask_dir": input_dir.with_name(f"{input_dir.name}_masks"),
        "source_id": source_id,
        **overrides,
    }
    write_index_config(config_path, **values)
    return runner.invoke(app, ["index-data", "--config", str(config_path)])


def read_raw_manifest(tmp_path: Path) -> dict:
    manifest_path = tmp_path / "dataset" / "manifests" / "raw.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_index_data_rejects_missing_mask_dir(tmp_path: Path) -> None:
    result = run_index_data(tmp_path, tmp_path, mask_dir=tmp_path / "missing_masks")

    assert result.exit_code != 0
    assert "Required mask_dir directory not found" in result.output
    assert "Configuration validation failed" not in result.output


def test_index_data_rejects_unmatched_subjects_when_masks_required(
    tmp_path: Path,
) -> None:
    volumes = create_volume_source(tmp_path, "volumes", ["subject_1"])
    create_volume_source(tmp_path, "other", ["subject_2"])

    result = run_index_data(tmp_path, volumes, mask_dir=tmp_path / "other_masks")

    assert result.exit_code != 0
    assert "Volume and mask subject IDs do not match" in result.output


def test_index_data_appends_sources_and_namespaces_colliding_subject_ids(
    tmp_path: Path,
) -> None:
    first = create_volume_source(tmp_path, "first", ["subject_1", "subject_2"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])

    assert run_index_data(tmp_path, first, "siteA").exit_code == 0
    result = run_index_data(tmp_path, second, "siteB")

    assert result.exit_code == 0, result.output
    manifest = read_raw_manifest(tmp_path)
    assert [source["source_id"] for source in manifest["sources"]] == [
        "siteA",
        "siteB",
    ]
    assert [subject["subject_id"] for subject in manifest["subjects"]] == [
        "siteA_subject_1",
        "siteA_subject_2",
        "siteB_subject_1",
    ]
    assert manifest["created_at"] <= manifest["updated_at"]


def test_index_data_rejects_duplicate_source_id_and_keeps_manifest(
    tmp_path: Path,
) -> None:
    first = create_volume_source(tmp_path, "first", ["subject_1"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])
    assert run_index_data(tmp_path, first).exit_code == 0

    result = run_index_data(tmp_path, second)

    assert result.exit_code != 0
    assert "Source ID already exists" in result.output
    assert len(read_raw_manifest(tmp_path)["sources"]) == 1


def test_preprocess_rejects_missing_dataset_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocess.toml"
    config_path.write_text(
        f"dataset_dir = {json.dumps(str(tmp_path / 'missing'))}\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["preprocess", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "Configuration validation failed" in result.output
