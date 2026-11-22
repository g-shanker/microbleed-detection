import numpy as np
import pytest

from microbleednet.core.transforms import patch
from microbleednet.core.utils import label_components
from tests.support_core import voxels


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (
            voxels((3, 3, 3), (0, 0, 0), (2, 2, slice(1, None)), dtype=np.uint8),
            voxels((3, 3, 3), (0, 0, 0), dtype=int)
            + voxels((3, 3, 3), (2, 2, slice(1, None)), value=2, dtype=int),
        ),
        (np.array([[[0.2, 0.8]]]) > 0.5, np.array([[[0, 1]]])),
    ],
    ids=["two-components", "thresholded-probabilities"],
)
def test_label_components_returns_connected_labels(
    mask: np.ndarray, expected: np.ndarray
) -> None:
    np.testing.assert_array_equal(label_components(mask, 3), expected)


def test_get_target_centers_returns_rounded_centers() -> None:
    labels = voxels((3, 3, 3), (0, 0, 0), dtype=np.int32)
    labels[2, 2, 1:] = 2

    assert patch.get_target_centers(labels) == [(0, 0, 0), (2, 2, 2)]


def test_nonoverlapping_patches_pad_each_axis_to_patch_size_multiple() -> None:
    volume = np.arange(30 * 24 * 24).reshape(30, 24, 24)

    patches = patch.get_nonoverlapping_patches(volume, patch_size=24)

    assert [item.shape for item in patches] == [(24, 24, 24)] * 2
    np.testing.assert_array_equal(patches[0], volume[:24])
    np.testing.assert_array_equal(patches[1][:6], volume[24:])
    np.testing.assert_array_equal(patches[1][6:], 0)


def test_nonoverlapping_patches_pad_volume_smaller_than_patch() -> None:
    patches = patch.get_nonoverlapping_patches(np.ones((2, 3, 4)), patch_size=5)

    assert len(patches) == 1
    assert patches[0].shape == (5, 5, 5)
    np.testing.assert_array_equal(patches[0][:2, :3, :4], 1)


def test_overlapping_patch_bounds_cover_each_axis_and_trailing_edge() -> None:
    bounding_boxes = patch.get_overlapping_patch_bounding_boxes(
        shape=(3, 6, 5), patch_size=4, overlap=2
    )

    assert bounding_boxes == [
        ((0, 3), (0, 4), (0, 4)),
        ((0, 3), (2, 6), (0, 4)),
        ((0, 3), (0, 4), (1, 5)),
        ((0, 3), (2, 6), (1, 5)),
    ]


def test_centered_patches_use_requested_size_at_volume_boundary() -> None:
    patches = patch.extract_centered_patches(
        np.ones((4, 4, 4)), centers=[(0, 0, 0)], patch_size=4
    )

    assert patches[0].shape == (4, 4, 4)
    np.testing.assert_array_equal(patches[0][2:, 2:, 2:], 1)


def test_centered_patches_place_center_at_patch_center() -> None:
    patches = patch.extract_centered_patches(
        voxels((7, 7, 7), (3, 3, 3)), centers=[(3, 3, 3)], patch_size=5
    )

    assert patches[0][2, 2, 2] == 1
    assert np.count_nonzero(patches[0]) == 1
