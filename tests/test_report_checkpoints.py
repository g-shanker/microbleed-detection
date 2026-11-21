import json
import logging
from pathlib import Path

import pytest

from microbleednet import report_checkpoints
from microbleednet.errors import ApplicationError

SETTINGS = {"radii": (1.0, 2.0), "subjects": ["a", "b"]}


def test_append_then_read_round_trips_subjects(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"

    report_checkpoints.append(path, SETTINGS, "a", [{"value": 1}])
    report_checkpoints.append(path, SETTINGS, "b", [{"value": float("nan")}])

    finished = report_checkpoints.read(path, SETTINGS)
    assert list(finished) == ["a", "b"]
    assert finished["a"] == [{"value": 1}]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_missing_or_empty_checkpoints_hold_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    report_checkpoints.append(None, SETTINGS, "a", [])

    assert report_checkpoints.read(None, SETTINGS) == {}
    assert report_checkpoints.read(tmp_path / "missing.jsonl", SETTINGS) == {}
    assert report_checkpoints.read(empty, SETTINGS) == {}


def test_read_rejects_different_settings(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    report_checkpoints.append(path, SETTINGS, "a", [])

    with pytest.raises(ApplicationError, match="different settings"):
        report_checkpoints.read(path, {**SETTINGS, "subjects": ["a"]})


def test_read_skips_a_cut_off_last_line(tmp_path: Path, caplog) -> None:
    path = tmp_path / "checkpoint.jsonl"
    report_checkpoints.append(path, SETTINGS, "a", [{"value": 1}])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"subject_id": "b", "rows": []})[:10])

    with caplog.at_level(logging.WARNING):
        finished = report_checkpoints.read(path, SETTINGS)

    assert list(finished) == ["a"]
    assert "Ignoring incomplete checkpoint line 3" in caplog.text
