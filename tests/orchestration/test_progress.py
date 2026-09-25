from microbleednet.progress import Progress, progress


def test_progress_uses_a_silent_tracker_by_default() -> None:
    items = [1, 2, 3]

    assert Progress().track(items, "ignored") is items


def test_progress_can_configure_a_tracker() -> None:
    progress_instance = Progress()
    tracked_items = [1, 2, 3]
    descriptions: list[str] = []

    def tracker(items, description):
        descriptions.append(description)
        return items

    assert progress_instance.track(tracked_items, "before") is tracked_items
    progress_instance.configure(tracker)
    assert progress_instance.track(tracked_items, "after") is tracked_items
    assert descriptions == ["after"]
    assert isinstance(progress, Progress)
