from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast


def format_validation_errors(
    errors: Iterable[Mapping[str, object]],
) -> str:
    """Format structured validation errors as concise field-level details."""
    return "; ".join(
        f"{'.'.join(str(part) for part in cast(Iterable[object], item['loc']))}: "
        f"{item['msg']}"
        for item in errors
    )


@dataclass
class ApplicationError(ValueError):
    """An expected failure with actionable information for the CLI."""

    category: str
    summary: str
    cause: str | None = None
    fix: str | None = None
    context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the base exception with the error summary."""
        Exception.__init__(self, self.summary)

    def __str__(self) -> str:
        """Combine the summary, cause, and fix into a readable message."""
        parts = [self.summary]
        if self.cause:
            parts.append(f"Cause: {self.cause}")
        if self.fix:
            parts.append(f"Fix: {self.fix}")
        return ". ".join(parts) + "."


def print_application_error(error: ApplicationError) -> None:
    """Print an application error without requiring a presentation framework."""
    message = str(error)
    if error.context:
        context = ". ".join(
            f"{key}: {value}" for key, value in error.context.items()
        )
        message = f"{message} Context: {context}."
    print(message)


ErrorRenderFunction = Callable[[ApplicationError], None]


class ErrorRenderer:
    """Dispatch application errors through the configured presentation layer."""

    def __init__(self) -> None:
        """Default error rendering to the presentation-neutral printer."""
        self._render: ErrorRenderFunction = print_application_error

    def configure(self, renderer: ErrorRenderFunction) -> None:
        """Replace the active application-error renderer."""
        self._render = renderer

    def render(self, error: ApplicationError) -> None:
        """Render an application error with the configured renderer."""
        self._render(error)


error_renderer = ErrorRenderer()
