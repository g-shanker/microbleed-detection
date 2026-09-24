import numpy as np

from microbleednet.core.transforms import patch
from microbleednet.core.utils import label_components


def test_label_components_returns_connected_labels() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[2, 2, 1:] = 1

    labels = label_components(mask, 3)

    assert labels[0, 0, 0] == 1
    assert labels[2, 2, 1] == 2


def test_label_components_labels_explicitly_thresholded_mask() -> None:
    probability_map = np.array([[[0.2, 0.8]]])

    labels = label_components(probability_map > 0.5, 3)

    np.testing.assert_array_equal(labels, np.array([[[0, 1]]]))


def test_get_target_centers_returns_rounded_centers() -> None:
    labels = np.zeros((3, 3, 3), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[2, 2, 1:] = 2

    centers = patch.get_target_centers(labels)

    assert centers == [(0, 0, 0), (2, 2, 2)]


def test_nonoverlapping_patches_use_requested_size_and_align_final_window() -> None:
    volume = np.arange(30 * 24 * 24).reshape(30, 24, 24)

    patches = patch.get_nonoverlapping_patches(volume, patch_size=24)

    assert [item.shape for item in patches] == [(24, 24, 24)] * 2
    np.testing.assert_array_equal(patches[0], volume[:24])
    np.testing.assert_array_equal(patches[1], volume[-24:])


def test_nonoverlapping_patches_pad_volume_smaller_than_patch() -> None:
    patches = patch.get_nonoverlapping_patches(np.ones((2, 3, 4)), patch_size=5)

    assert len(patches) == 1
    assert patches[0].shape == (5, 5, 5)
    np.testing.assert_array_equal(patches[0][:2, :3, :4], 1)


def test_centered_patches_use_requested_size_at_volume_boundary() -> None:
    volume = np.ones((4, 4, 4))

    patches = patch.extract_centered_patches(
        volume,
        centers=[(0, 0, 0)],
        patch_size=4,
    )

    assert patches[0].shape == (4, 4, 4)
    np.testing.assert_array_equal(patches[0][2:, 2:, 2:], 1)


def test_centered_patches_place_center_at_patch_center() -> None:
    volume = np.zeros((7, 7, 7))
    volume[3, 3, 3] = 1

    patches = patch.extract_centered_patches(
        volume,
        centers=[(3, 3, 3)],
        patch_size=5,
    )

    assert patches[0][2, 2, 2] == 1
    assert np.count_nonzero(patches[0]) == 1