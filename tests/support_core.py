"""Array and image builders shared by core engine and transform tests."""

from typing import Any

import nibabel as nib
import numpy as np

from tests.support import DEFAULT_SHAPE, nifti

Index = tuple[int | slice, ...]


def voxels(
    shape: tuple[int, ...], *indices: Index, value: float = 1, dtype: Any = float
) -> np.ndarray:
    """Build a zero array with ``value`` written at every given index or slice."""
    array = np.zeros(shape, dtype=dtype)
    for index in indices:
        array[index] = value
    return array


def image(
    shape: tuple[int, ...] = DEFAULT_SHAPE, value: float = 1.0, **kwargs: Any
) -> nib.Nifti1Image:
    """Build a float NIfTI image; ``support.nifti`` fills int64 data by default."""
    kwargs.setdefault("dtype", float)
    return nifti(shape, value=value, **kwargs)
