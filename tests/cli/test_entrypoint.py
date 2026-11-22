import logging
from unittest.mock import Mock

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from microbleednet.cli import entrypoint
from microbleednet.errors import ApplicationError
from microbleednet.progress import progress, silent_track
from tests.support_cli_core import cli_globals_restored  # noqa: F401

pytestmark = pytest.mark.usefixtures("cli_globals_restored")

runner = CliRunner()


@pytest.mark.parametrize(
    ("flags", "level", "silent"),
    [
        ({}, logging.INFO, False),
        ({"verbose": True}, logging.DEBUG, False),
        ({"quiet": True}, logging.WARNING, True),
    ],
)
def test_configure_cli_sets_log_level_and_progress(
    flags: dict[str, bool], level: int, silent: bool
) -> None:
    entrypoint.configure_cli(**{"quiet": False, "verbose": False, **flags})

    assert logging.getLogger().level == level
    assert (progress._track is silent_track) is silent
    assert list(progress.track([1, 2], "test")) == [1, 2]


def test_quiet_and_verbose_are_mutually_exclusive() -> None:
    with pytest.raises(typer.BadParameter, match="conflicting log levels"):
        entrypoint.configure_cli(quiet=True, verbose=True)


@pytest.mark.parametrize(
    ("flag", "level"), [("--verbose", logging.DEBUG), ("--quiet", logging.WARNING)]
)
def test_cli_flags_reach_configure_cli(flag: str, level: int) -> None:
    result = runner.invoke(entrypoint.app, [flag, "index-data", "--help"])

    assert result.exit_code == 0, result.output
    assert logging.getLogger().level == level


def test_render_application_error_supports_summary_only() -> None:
    console = Console(record=True)

    entrypoint.render_application_error(
        ApplicationError(category="Internal", summary="Operation failed"),
        target_console=console,
    )

    output = console.export_text()
    assert "Internal error" in output
    assert "Operation failed" in output


def test_render_application_error_includes_traceback_in_verbose_mode() -> None:
    console = Console(record=True)
    error = ApplicationError(
        category="Input data",
        summary="Input data is invalid",
        cause="The test fixture is incomplete.",
        fix="Provide the missing input.",
    )
    error.__cause__ = RuntimeError("root failure")

    entrypoint.render_application_error(error, target_console=console, verbose=True)

    output = console.export_text()
    assert "Details:" in output
    assert "RuntimeError: root failure" in output


def test_main_runs_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = Mock()
    monkeypatch.setattr(entrypoint, "app", cli)

    entrypoint.main()

    cli.assert_called_once_with()
