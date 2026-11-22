from microbleednet.progress import Progress, progress


def test_progress_uses_a_silent_tracker_by_default() -> None:
    items = [1, 2, 3]

    assert Progress().track(items, "ignored") is items


def test_progress_can_configure_a_tracker() -> None:
    instance = Progress()
    items = [1, 2, 3]
    descriptions: list[str] = []

    def tracker(tracked, description):
        descriptions.append(description)
        return tracked

    instance.configure(tracker)

    assert instance.track(items, "after") is items
    assert descriptions == ["after"]


def test_shared_progress_is_a_progress_instance() -> None:
    assert isinstance(progress, Progress)
