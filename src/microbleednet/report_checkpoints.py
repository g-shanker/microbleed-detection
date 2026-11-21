"""Append-only per-subject checkpoints shared by the diagnostic reports.

Experimental. See EXPERIMENTAL_CHANGES.md for how to remove it.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .errors import ApplicationError

logger = logging.getLogger(__name__)


def read(path: Path | None, settings: dict[str, Any]) -> dict[str, list[Any]]:
    """Return finished subjects' rows, checking the run settings match."""
    if path is None or not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    if json.loads(lines[0]) != {"settings": normalized(settings)}:
        raise ApplicationError(
            category="Configuration",
            summary="Checkpoint was written with different settings",
            cause=f"'{path}' does not match the current arguments",
            fix="Rerun with the original arguments or choose a new checkpoint file",
            context={"path": str(path)},
        )
    subjects: dict[str, list[Any]] = {}
    for number, line in enumerate(lines[1:], start=2):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # A crash can cut off the last line; that subject is redone.
            logger.warning("Ignoring incomplete checkpoint line %d", number)
            continue
        subjects[entry["subject_id"]] = entry["rows"]
    return subjects


def append(
    path: Path | None, settings: dict[str, Any], subject_id: str, rows: list[Any]
) -> None:
    """Append one finished subject, writing the settings header first."""
    if path is None:
        return
    start = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8") as handle:
        if start:
            handle.write(json.dumps({"settings": normalized(settings)}) + "\n")
        handle.write(json.dumps({"subject_id": subject_id, "rows": rows}) + "\n")


def normalized(settings: dict[str, Any]) -> Any:
    """Return settings as they read back from JSON, so tuples compare equal."""
    return json.loads(json.dumps(settings))
