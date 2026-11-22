"""Write the masks and volumes of a run as NIfTI files, to diagnose failures.

Layout under ``<output_dir>/masks/<subject_id>/``::

    volume.nii.gz            volume sent to vessel inpainting
    lesions.nii.gz           annotated lesions
    brain.nii.gz             brain mask
    <strategy>_n<N>.nii.gz   dilated vessel mask that was inpainted
    <strategy>_n<N>_inpainted.nii.gz   result of inpainting (optional)

All files share the geometry of the preprocessed volume, so they overlay in any
NIfTI viewer.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from .prepare import PreparedSubject


def subject_directory(output_dir: Path, subject_id: str) -> Path:
    return output_dir / "masks" / subject_id


def mask_name(strategy: str, iterations: int) -> str:
    return f"{strategy}_n{iterations}"


def save_subject_inputs(prepared: PreparedSubject, directory: Path) -> None:
    """Write the volume, lesion mask and brain mask of one subject."""
    save(directory / "volume.nii.gz", prepared.volume.astype(np.float32), prepared)
    save(directory / "lesions.nii.gz", prepared.lesions, prepared)
    save(directory / "brain.nii.gz", prepared.brain, prepared)


def save_outcome(
    prepared: PreparedSubject,
    directory: Path,
    strategy: str,
    iterations: int,
    mask: np.ndarray,
    inpainted: np.ndarray | None,
) -> None:
    """Write one strategy and iteration count's mask and, optionally, its volume."""
    name = mask_name(strategy, iterations)
    save(directory / f"{name}.nii.gz", mask, prepared)
    if inpainted is not None:
        save(
            directory / f"{name}_inpainted.nii.gz",
            inpainted.astype(np.float32),
            prepared,
        )


def save(path: Path, array: np.ndarray, prepared: PreparedSubject) -> None:
    """Save an array with the subject's geometry; booleans are stored as uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array.astype(np.uint8) if array.dtype == bool else array
    nib.save(nib.Nifti1Image(data, prepared.affine), path)
