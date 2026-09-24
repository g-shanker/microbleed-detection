import logging
import traceback
from collections.abc import Iterable
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.progress import Progress as RichProgress

from ..errors import ApplicationError, error_renderer
from ..progress import ProgressItem, progress, silent_track
from .commands import register_commands

console = Console()


def render_application_error(
    error: ApplicationError,
    target_console: Console | None = None,
    verbose: bool = False,
) -> None:
    """Render expected failures using the CLI's Rich presentation settings."""
    if target_console is None:
        target_console = console
    lines = [error.summary]
    if error.cause:
        lines.append(f"Cause: {error.cause}")
    if error.fix:
        lines.append(f"Fix: {error.fix}")
    lines.extend(f"{key}: {value}" for key, value in error.context.items())
    if verbose:
        lines.append("Details:")
        lines.append("".join(traceback.format_exception(error)).rstrip())
    target_console.print(
        Panel(
            "\n".join(lines),
            title=f"{error.category} error",
            border_style="red",
        )
    )


app = typer.Typer(
    name="microbleednet",
    help="Run a microbleednet pipeline command.",
    no_args_is_help=True,
)

register_commands(app, console)


@app.callback()
def configure_cli(
    verbose: Annotated[bool, typer.Option(help="Emit debug-level logs.")] = False,
    quiet: Annotated[bool, typer.Option(help="Only emit warnings and errors.")] = False,
) -> None:
    """Configure Rich logging once for the whole CLI."""
    if verbose and quiet:
        raise typer.BadParameter(
            "--verbose and --quiet select conflicting log levels; choose one"
        )

    def rich_render_application_error(error: ApplicationError) -> None:
        render_application_error(error, verbose=verbose)

    error_renderer.configure(rich_render_application_error)

    def rich_track(
        items: Iterable[ProgressItem],
        description: str,
    ) -> Iterable[ProgressItem]:
        with RichProgress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as rich_progress:
            yield from rich_progress.track(items, description=description)

    if verbose:
        level = logging.DEBUG
        tracker = rich_track
    elif quiet:
        level = logging.WARNING
        tracker = silent_track
    else:
        level = logging.INFO
        tracker = rich_track

    progress.configure(tracker)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
        force=True,
    )


def main() -> None:
    app()
