"""Progress logging for the experiment run."""

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

Log = Callable[[str], None]


def silent(message: str) -> None:
    """Discard a message; the default when a caller does not want progress."""


class RunLog:
    """Print progress with a time of day and append the same text to a file.

    Calling the log writes one timestamped line. ``raw`` writes text as is, for
    tables. The file is opened in append mode, so reruns add to the same
    ``run.log`` after a header line.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(f"=== run started {datetime.now():%Y-%m-%d %H:%M:%S} ===")

    def __call__(self, message: str) -> None:
        self._write(f"[{datetime.now():%H:%M:%S}] {message}")

    def raw(self, text: str) -> None:
        self._write(text)

    def _write(self, text: str) -> None:
        print(text, file=sys.stdout, flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text + "\n")
