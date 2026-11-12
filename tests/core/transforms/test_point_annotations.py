import numpy as np

from microbleednet.core.transforms.point_annotations import dilate_point_annotations


def test_dilates_single_point_into_physical_ball() -> None:
    mask = np.zeros((21, 21, 21), dtype=np.uint8)
    mask[10, 10, 10] = 1

    dilated = dilate_point_annotations(mask, (1.0, 1.0, 1.0), 2.0)

    coords = np.argwhere(dilated)
    distances = np.linalg.norm(coords - 10, axis=1)
    assert dilated.dtype == np.uint8
    assert distances.max() <= 2.0
    assert int(dilated.sum()) == 33


def test_respects_anisotropic_spacing() -> None:
    mask = np.zeros((21, 21, 11), dtype=np.uint8)
    mask[10, 10, 5] = 1

    dilated = dilate_point_annotations(mask, (0.5, 0.5, 2.0), 2.0)

    coords = np.argwhere(dilated)
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    assert extent.tolist() == [9, 9, 3]


def test_two_slice_point_becomes_a_single_ball() -> None:
    single = np.zeros((21, 21, 21), dtype=np.uint8)
    single[10, 10, 10] = 1
    two_slice = single.copy()
    two_slice[10, 10, 11] = 1

    dilated_single = dilate_point_annotations(single, (1.0, 1.0, 1.0), 2.0)
    dilated_two_slice = dilate_point_annotations(two_slice, (1.0, 1.0, 1.0), 2.0)

    assert int(dilated_two_slice.sum()) == int(dilated_single.sum())


def test_keeps_original_points_and_separate_lesions() -> None:
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    mask[5, 5, 5] = 1
    mask[20, 20, 20] = 1

    dilated = dilate_point_annotations(mask, (1.0, 1.0, 1.0), 2.5)

    assert np.all(dilated[mask > 0] == 1)
    assert int(dilated[:12].sum()) == int(dilated[12:].sum())


def test_clips_points_at_volume_edge() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0, 9, 0] = 1

    dilated = dilate_point_annotations(mask, (1.0, 1.0, 1.0), 2.0)

    assert dilated[0, 9, 0] == 1
    assert dilated.shape == mask.shape


def test_empty_mask_stays_empty() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)

    dilated = dilate_point_annotations(mask, (1.0, 1.0, 1.0), 2.0)

    assert not dilated.any()
