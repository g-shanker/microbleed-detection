import json
from pathlib import Path

import pytest

from microbleednet.orchestration import atomic_io
from microbleednet.orchestration.atomic_io import read_json, write_json_atomic


def test_write_json_atomic_creates_parent_and_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"

    write_json_atomic(path, {"zebra": 1, "apple": "microbleed"})

    content = path.read_text(encoding="utf-8")
    assert content == '{\n  "apple": "microbleed",\n  "zebra": 1\n}\n'


def test_write_json_atomic_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("old content", encoding="utf-8")

    write_json_atomic(path, ["new", "content"])

    assert read_json(path) == ["new", "content"]
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_removes_temporary_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "manifest.json"

    with pytest.raises(TypeError):
        write_json_atomic(path, {"invalid": object()})

    assert list(path.parent.glob("tmp*")) == []
    assert not path.exists()


def test_write_json_atomic_handles_temporary_file_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    def raise_creation_error(*args: object, **kwargs: object) -> None:
        raise OSError("could not create temporary file")

    monkeypatch.setattr(
        atomic_io.tempfile, "NamedTemporaryFile", raise_creation_error
    )

    with pytest.raises(OSError, match="could not create temporary file"):
        write_json_atomic(path, {"key": "value"})

    assert not path.exists()


def test_read_json_rejects_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json(path)
