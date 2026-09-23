"""Helpers shared by the CLI commands.

The commands are generated from a declarative table in ``cli/commands.py`` and
wired up by ``entrypoint.py``. Everything they need but do not own — config
loading and parsing, config-key introspection, and output — lives here.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Literal,
    Protocol,
    TypeGuard,
    get_args,
    get_origin,
)

import typer
from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class CommandSpec:
    """Everything that distinguishes one CLI command from the shared skeleton."""

    name: str
    help: str
    config: type[BaseModel]
    pipe: str  # module under orchestration.pipes; imported lazily to defer torch


class PipeModule(Protocol):
    """Interface required by a lazily loaded orchestration pipe."""

    def execute(self, config: BaseModel) -> None: ...


def is_pipe_module(module: ModuleType) -> TypeGuard[PipeModule]:
    """Return whether a loaded module exposes a callable execute function."""
    execute = getattr(module, "execute", None)
    return callable(execute)


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file into a plain dict."""
    if not path.is_file():
        raise ValueError(f"configuration file does not exist: {path}")
    if path.suffix.lower() != ".toml":
        raise ValueError(f"configuration file must be a .toml file: {path}")
    try:
        with path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read configuration {path}: {error}") from error
    return config


def parse_config[ConfigModel: BaseModel](
    path: Path, model: type[ConfigModel]
) -> ConfigModel:
    """Load a config file and validate it into a typed model.

    Pydantic validation errors are surfaced as clean CLI errors rather than
    tracebacks.
    """
    try:
        return model.model_validate(load_config(path))
    except ValidationError as error:
        raise typer.BadParameter(f"invalid configuration {path}:\n{error}") from error


@dataclass(frozen=True)
class ConfigField:
    """A single leaf configuration key derived from a config model."""

    section: str
    key: str
    default: str | None
    description: str


def config_fields(model: type[BaseModel], prefix: str) -> list[ConfigField]:
    """Flatten a config model (recursing into nested models) into leaf fields.

    Derived from the model so descriptions, defaults, and required-ness never
    drift from the Pydantic configuration models.
    """
    fields: list[ConfigField] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        annotation = field.annotation
        nested_model = None
        candidates = (annotation, *get_args(annotation))
        for candidate in candidates:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                nested_model = candidate
                break
        if nested_model is not None:
            fields.extend(config_fields(nested_model, prefix=f"{key}."))
            continue
        section, _, leaf = key.rpartition(".")
        default = None if field.is_required() else repr(field.default)
        description = field.description or ""
        if get_origin(annotation) is Literal:
            choices = ", ".join(repr(choice) for choice in get_args(annotation))
            description = f"{description} Allowed values: {choices}."
        fields.append(
            ConfigField(
                section=section,
                key=leaf,
                default=default,
                description=description,
            )
        )
    return fields


def describe_hint(command: str) -> str:
    """One-line epilog pointing users at the full config reference."""
    return f"Run 'microbleednet describe {command}' for the configuration keys."
