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
    return items


class Progress:
    def __init__(self) -> None:
        self._track: ProgressTracker = silent_track

    def configure(self, tracker: ProgressTracker) -> None:
        self._track = tracker

    def track(
        self,
        items: Iterable[ProgressItem],
        description: str,
    ) -> Iterable[ProgressItem]:
        return self._track(items, description)


progress = Progress()