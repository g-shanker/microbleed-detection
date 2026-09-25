from collections.abc import Iterable
from typing import Callable, TypeVar

ProgressItem = TypeVar("ProgressItem")

ProgressTracker = Callable[
    [Iterable[ProgressItem], str],
    Iterable[ProgressItem],
]


def silent_track(
    items: Iterable[ProgressItem],
    description: str,
) -> Iterable[ProgressItem]:
    """Return items unchanged when progress display is disabled."""
    return items


class Progress:
    def __init__(self) -> None:
        """Initialize progress tracking with the silent tracker."""
        self._track: ProgressTracker = silent_track

    def configure(self, tracker: ProgressTracker) -> None:
        """Replace the active progress-tracking strategy."""
        self._track = tracker

    def track(
        self,
        items: Iterable[ProgressItem],
        description: str,
    ) -> Iterable[ProgressItem]:
        """Track items with the configured progress strategy."""
        return self._track(items, description)


progress = Progress()