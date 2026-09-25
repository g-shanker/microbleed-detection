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

from pydantic import BaseModel, ValidationError

from ..errors import ApplicationError, format_validation_errors


@dataclass(frozen=True)
class CommandSpec:
    """Everything that distinguishes one CLI command from the shared skeleton."""

    name: str
    help: str
    config: type[BaseModel]
    pipe: str  # module under orchestration.pipes; imported lazily to defer torch


class PipeModule(Protocol):
    """Interface required by a lazily loaded orchestration pipe."""

    def execute(self, config: BaseModel) -> None:
        """Run the pipe with a validated configuration model."""
        ...


def is_pipe_module(module: ModuleType) -> TypeGuard[PipeModule]:
    """Return whether a loaded module exposes a callable execute function."""
    execute = getattr(module, "execute", None)
    return callable(execute)


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file into a plain dict."""
    if not path.is_file():
        raise ApplicationError(
            category="Configuration",
            summary="Configuration file not found",
            fix="Provide an existing TOML file with --config",
            context={"path": str(path)},
        )
    if path.suffix.lower() != ".toml":
        raise ApplicationError(
            category="Configuration",
            summary="Configuration file must use the TOML format",
            cause=f"Received suffix '{path.suffix or '<none>'}'",
            fix="Use a file with a .toml extension",
            context={"path": str(path)},
        )
    try:
        with path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, ValueError) as error:
        raise ApplicationError(
            category="Configuration",
            summary="Could not load configuration file",
            cause=str(error).rstrip("."),
            fix="Correct the file permissions or TOML syntax",
            context={"path": str(path)},
        ) from error
    return config


def parse_config[ConfigModel: BaseModel](
    path: Path, model: type[ConfigModel], command: str
) -> ConfigModel:
    """Load a config file and validate it into a typed model.

    Pydantic validation errors become structured application errors.
    """
    try:
        return model.model_validate(load_config(path))
    except ValidationError as error:
        validation_errors = error.errors()
        if len(validation_errors) == 1:
            nested_error = validation_errors[0].get("ctx", {}).get("error")
            if isinstance(nested_error, ApplicationError):
                raise nested_error from error

        details = format_validation_errors(validation_errors)
        raise ApplicationError(
            category="Configuration",
            summary="Configuration validation failed",
            cause=details,
            fix=(
                f"Correct the reported fields; run 'microbleednet describe {command}' "
                "for valid configuration keys"
            ),
            context={"path": str(path), "model": model.__name__},
        ) from error


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
