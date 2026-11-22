from pathlib import Path

import numpy as np

from experiments.inpainting import export
from experiments.inpainting.prepare import PreparedSubject
from experiments.inpainting.tests.support import AFFINE, LESION, SHAPE, base_volume
from microbleednet.core.io import load_volume


def prepared() -> PreparedSubject:
    lesions = np.zeros(SHAPE, dtype=bool)
    lesions[LESION] = True
    return PreparedSubject(
        volume=base_volume(),
        lesions=lesions,
        brain=np.ones(SHAPE, dtype=bool),
        affine=AFFINE,
        spacing=(1.0, 1.0, 3.0),
        vessels={},
    )


def test_masks_live_under_a_directory_per_subject(tmp_path: Path) -> None:
    assert export.subject_directory(tmp_path, "s1") == tmp_path / "masks" / "s1"


def test_files_are_named_after_strategy_and_iterations() -> None:
    assert export.mask_name("frangi3d", 3) == "frangi3d_n3"


def test_subject_inputs_are_saved_with_the_subjects_geometry(tmp_path: Path) -> None:
    export.save_subject_inputs(prepared(), tmp_path / "s1")

    for name in ("volume", "lesions", "brain"):
        image = load_volume(tmp_path / "s1" / f"{name}.nii.gz")
        np.testing.assert_array_equal(image.affine, AFFINE)
        assert image.shape == SHAPE


def test_masks_are_stored_as_integers_and_the_volume_as_float(tmp_path: Path) -> None:
    export.save_subject_inputs(prepared(), tmp_path)

    lesions = load_volume(tmp_path / "lesions.nii.gz")
    assert lesions.get_data_dtype() == np.uint8
    assert np.asarray(lesions.dataobj)[LESION] == 1
    assert load_volume(tmp_path / "volume.nii.gz").get_data_dtype() == np.float32


def test_an_outcome_saves_its_mask(tmp_path: Path) -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1, 2, 1] = True

    export.save_outcome(prepared(), tmp_path, "axial", 2, mask, None)

    saved = np.asarray(load_volume(tmp_path / "axial_n2.nii.gz").dataobj)
    np.testing.assert_array_equal(saved, mask.astype(np.uint8))
    assert not (tmp_path / "axial_n2_inpainted.nii.gz").exists()


def test_an_outcome_can_also_save_the_inpainted_volume(tmp_path: Path) -> None:
    inpainted = base_volume() + 1

    export.save_outcome(
        prepared(), tmp_path, "frangi3d", 0, np.zeros(SHAPE, bool), inpainted
    )

    saved = load_volume(tmp_path / "frangi3d_n0_inpainted.nii.gz")
    np.testing.assert_allclose(np.asarray(saved.dataobj), inpainted, rtol=1e-6)
    assert saved.get_data_dtype() == np.float32
