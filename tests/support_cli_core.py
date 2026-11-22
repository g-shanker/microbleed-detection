"""Helpers shared by the CLI and core test modules."""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import BaseModel, Field

from microbleednet.cli.utils import CommandSpec
from microbleednet.core.datamodels import PatchRecord
from microbleednet.errors import error_renderer
from microbleednet.progress import progress


class Nested(BaseModel):
    knob: int = Field(default=1, description="A nested knob.")


class FakeConfig(BaseModel):
    top_level: str = Field(description="A top-level field.")
    nested: Nested = Field(default_factory=Nested)
    trailing: bool = Field(default=True, description="A trailing top-level field.")


FAKE_SPEC = CommandSpec(
    name="fake", help="Fake command for testing.", config=FakeConfig, pipe="unused"
)


@pytest.fixture
def cli_globals_restored() -> Iterator[None]:
    """Undo the logging, progress and error-renderer state set by ``configure_cli``."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    tracker, renderer = progress._track, error_renderer._render
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    progress.configure(tracker)
    error_renderer.configure(renderer)


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare tensors with ``torch.testing.assert_close`` tolerances for float32.

    ``torch.testing.assert_close`` costs about two seconds on its first call.
    """
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1.3e-6, atol=1e-5)


def patch_record(
    directory: Path | None = None, *, positive: bool = True
) -> PatchRecord:
    """Build a patch record, writing real 4x4x4 arrays when ``directory`` is given.

    The arrays hold a volume of ones, a mask of ones and an FRST map of twos.
    """
    arrays = {
        "volume": np.ones((1, 4, 4, 4)),
        "mask": np.ones((1, 4, 4, 4), dtype=np.uint8),
        "frst": np.full((1, 4, 4, 4), 2.0),
    }
    paths = {name: name for name in arrays}
    if directory is not None:
        paths = {name: str(directory / f"{name}.npy") for name in arrays}
        for name, array in arrays.items():
            np.save(paths[name], array)
    return PatchRecord(
        volume_path=paths["volume"],
        mask_path=paths["mask"],
        frst_path=paths["frst"],
        patch_index=0,
        has_microbleed=positive,
    )
