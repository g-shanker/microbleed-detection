import logging
from collections.abc import Iterable
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
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

from ..progress import ProgressItem, progress, silent_track
from .commands import register_commands

console = Console()


app = typer.Typer(
    name="microbleednet",
    help="TODO: write out a help message",
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
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")

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
