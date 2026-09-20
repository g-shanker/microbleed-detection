"""Table-driven construction of the CLI commands.

Every command is the same skeleton — parse a config, honor
``--dry-run``, defer-import one pipe, and run it — so each is described
declaratively by a :class:`CommandSpec` and built by :func:`build_command`
rather than hand-written in its own module. Only the per-command specifics
(config model, help text, messages, and which pipe to run)
vary, and those are exactly the spec's fields.

The pipe is imported lazily inside the command body, by module name, so
``--help`` and ``describe`` never pay to import torch.
"""

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Callable

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from ..orchestration.configs import (
    EvaluateConfig,
    IndexDataConfig,
    PreprocessConfig,
    SplitConfig,
    TrainConfig,
)
from .utils import (
    config_fields,
    describe_hint,
    parse_config,
    report,
)

COMMAND_HELP = "TODO: write a help message"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandSpec:
    """Everything that distinguishes one CLI command from the shared skeleton."""

    name: str
    help: str
    config: type[BaseModel]
    pipe: str  # module under orchestration.pipes; imported lazily to defer torch
    dry_run_message: Callable[[Any], str]


SPECS: list[CommandSpec] = [
    CommandSpec(
        name="index-data",
        help=COMMAND_HELP,
        config=IndexDataConfig,
        pipe="index_data",
        dry_run_message=lambda s: (
            f"Index configuration valid; manifests would be written under "
            f"{s.dataset_dir} (dry run)"
        ),
    ),
    CommandSpec(
        name="preprocess",
        help=COMMAND_HELP,
        config=PreprocessConfig,
        pipe="preprocess",
        dry_run_message=lambda s: (
            f"Preprocess configuration valid for {s.dataset_dir} (dry run)"
        ),
    ),
    CommandSpec(
        name="split",
        help=COMMAND_HELP,
        config=SplitConfig,
        pipe="split",
        dry_run_message=lambda s: (
            f"Split configuration valid; split manifest would be written under "
            f"{s.experiment_dir} (dry run)"
        ),
    ),
    CommandSpec(
        name="train",
        help=COMMAND_HELP,
        config=TrainConfig,
        pipe="train",
        dry_run_message=lambda s: (
            f"Training configuration valid; artifacts would be written under "
            f"{s.experiment_dir} (dry run)"
        ),
    ),
    CommandSpec(
        name="evaluate",
        help=COMMAND_HELP,
        config=EvaluateConfig,
        pipe="evaluate",
        dry_run_message=lambda s: (
            f"Evaluation configuration valid for "
            f"{s.explicit.output_dir if s.explicit else s.experiment.experiment_dir} "
            "(dry run)"
        ),
    ),
]


def build_command(app: typer.Typer, spec: CommandSpec) -> None:
    """Register ``spec`` as a command on ``app``.

    The generated command runs the shared skeleton; the pipe is imported by
    module name inside the body so torch stays out of ``--help``/``describe``.
    """

    @app.command(spec.name, help=spec.help, epilog=describe_hint(spec.name))
    def command(
        config_path: Annotated[Path, typer.Option(..., "--config")],
        dry_run: Annotated[
            bool, typer.Option(help="Validate configuration without writing outputs.")
        ] = False,
    ) -> None:
        started_at = perf_counter()
        config = parse_config(config_path, spec.config)
        logger.info(
            "Starting %s with config %s%s.",
            spec.name,
            config_path,
            " (dry run)" if dry_run else "",
        )
        if dry_run:
            report(spec.dry_run_message(config))
            logger.info(
                "Completed %s dry run in %.2fs.",
                spec.name,
                perf_counter() - started_at,
            )
            return
        # Imported lazily, by module name, so torch stays out of --help/describe.
        logger.debug("Loading pipe %s for command %s.", spec.pipe, spec.name)
        pipe = import_module(
            f"..orchestration.pipes.{spec.pipe}", package=__package__
        )
        logger.debug("Executing pipe %s for command %s.", spec.pipe, spec.name)
        pipe.execute(config)
        logger.info(
            "Completed %s in %.2fs.", spec.name, perf_counter() - started_at
        )


def build_describe_command(app: typer.Typer, specs: list[CommandSpec]) -> None:
    """Register the meta-command that documents each command's config keys.

    ``describe`` reads only the config models (via ``config_fields``), so it
    never imports a pipe and stays torch-free.
    """

    configs = {spec.name: spec.config for spec in specs}

    @app.command(
        "describe",
        help="Print the configuration keys, descriptions, and defaults for a command.",
        no_args_is_help=True,
    )
    def describe_command(
        command: Annotated[
            str,
            typer.Argument(help=f"Command to describe: {', '.join(configs)}."),
        ],
    ) -> None:
        model = configs.get(command)
        if model is None:
            raise typer.BadParameter(
                f"unknown command '{command}'; choose one of: {', '.join(configs)}"
            )

        table = Table(
            title=f"microbleednet {command} - configuration keys (TOML)",
            title_style="bold",
            header_style="bold",
            expand=True,
            row_styles=["", "dim"],
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Default", style="magenta", no_wrap=True)
        table.add_column("Description", ratio=1)

        current_section = ""
        for config_field in config_fields(model, prefix=""):
            if config_field.section != current_section:
                table.add_section()
                if config_field.section:
                    table.add_row(f"[bold]\\[{config_field.section}][/bold]", "", "")
                current_section = config_field.section
            required = config_field.default == "required"
            table.add_row(
                config_field.key,
                "[yellow]required[/yellow]" if required else config_field.default,
                config_field.description,
            )

        Console().print(table)


def register_commands(app: typer.Typer) -> None:
    """Register every pipe command and the ``describe`` meta-command."""
    for spec in SPECS:
        build_command(app, spec)
    build_describe_command(app, SPECS)
