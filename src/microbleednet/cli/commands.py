"""Table-driven construction of the CLI commands.

Every command is the same skeleton — parse a config, defer-import one pipe,
and run it — so each is described declaratively by a :class:`CommandSpec` and
built by :func:`build_command` rather than hand-written in its own module.
Only the per-command specifics (config model, help text, and which pipe to
run) vary, and those are exactly the spec's fields.

The pipe is imported lazily inside the command body, by module name, so
``--help`` and ``describe`` never pay to import torch.
"""

from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ..orchestration.configs import (
    EvaluateConfig,
    IndexDataConfig,
    InferConfig,
    PreprocessConfig,
    SplitConfig,
    TrainConfig,
)
from .utils import (
    CommandSpec,
    config_fields,
    describe_hint,
    is_pipe_module,
    parse_config,
)

COMMAND_HELP = "TODO: write a help message"


SPECS: list[CommandSpec] = [
    CommandSpec(
        name="index-data",
        help=COMMAND_HELP,
        config=IndexDataConfig,
        pipe="index_data",
    ),
    CommandSpec(
        name="preprocess",
        help=COMMAND_HELP,
        config=PreprocessConfig,
        pipe="preprocess",
    ),
    CommandSpec(
        name="split",
        help=COMMAND_HELP,
        config=SplitConfig,
        pipe="split",
    ),
    CommandSpec(
        name="train",
        help=COMMAND_HELP,
        config=TrainConfig,
        pipe="train",
    ),
    CommandSpec(
        name="evaluate",
        help=COMMAND_HELP,
        config=EvaluateConfig,
        pipe="evaluate",
    ),
    CommandSpec(
        name="infer",
        help=COMMAND_HELP,
        config=InferConfig,
        pipe="infer",
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
    ) -> None:
        config = parse_config(config_path, spec.config)

        # Imported lazily, by module name, so torch stays out of --help/describe.
        pipe = import_module(
            f"..orchestration.pipes.{spec.pipe}", package=__package__
        )
        if not is_pipe_module(pipe):
            raise TypeError(
                f"pipe module {spec.pipe!r} must define a callable execute(config)"
            )
        pipe.execute(config)


def build_describe_command(
    app: typer.Typer, specs: list[CommandSpec], console: Console
) -> None:
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
            table.add_row(
                config_field.key,
                    config_field.default
                    if config_field.default is not None
                    else "[yellow]required[/yellow]",
                config_field.description,
            )

        console.print(table)


def register_commands(app: typer.Typer, console: Console) -> None:
    """Register every pipe command and the ``describe`` meta-command."""
    for spec in SPECS:
        build_command(app, spec)
    build_describe_command(app, SPECS, console)
