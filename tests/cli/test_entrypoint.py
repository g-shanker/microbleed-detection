import logging
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from microbleednet.cli import entrypoint
from microbleednet.progress import progress, silent_track

app = entrypoint.app

runner = CliRunner()


def test_progress_reporter_is_always_available() -> None:
    entrypoint.configure_logging(quiet=False, verbose=False)

    assert progress._track is entrypoint.rich_track


def test_quiet_disables_progress() -> None:
    entrypoint.configure_logging(quiet=True, verbose=False)

    assert progress._track is silent_track
    assert logging.getLogger().level == logging.WARNING


def test_verbose_enables_debug_logging() -> None:
    entrypoint.configure_logging(quiet=False, verbose=True)

    assert logging.getLogger().level == logging.DEBUG


def test_quiet_and_verbose_are_mutually_exclusive() -> None:
    with pytest.raises(typer.BadParameter):
        entrypoint.configure_logging(quiet=True, verbose=True)


def test_cli_accepts_verbose_and_quiet_flags() -> None:
    for option in ("--verbose", "--quiet"):
        result = runner.invoke(app, [option, "index-data", "--help"])

        assert result.exit_code == 0, result.output


def test_main_runs_the_cli(monkeypatch) -> None:
    cli = Mock()
    monkeypatch.setattr(entrypoint, "app", cli)

    entrypoint.main()

    cli.assert_called_once_with()
