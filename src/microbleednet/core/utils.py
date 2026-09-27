import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label

from ..errors import ApplicationError
from .common.models import CandidateDetector, CandidateDiscriminatorTeacher


def fsl_executable(name: str) -> Path:
    """Return a validated executable path from the configured FSL installation."""
    fsldir_value = os.getenv("FSLDIR")
    if not fsldir_value:
        raise ApplicationError(
            category="Environment",
            summary="Image preprocessing requires FSL",
            cause="FSLDIR is not set",
            fix="Install FSL and set FSLDIR to its installation directory",
        )

    fsldir = Path(fsldir_value)
    if not fsldir.is_dir():
        raise ApplicationError(
            category="Environment",
            summary="Image preprocessing requires FSL",
            cause=f"FSLDIR does not reference a directory: '{fsldir}'",
            fix="Install FSL and set FSLDIR to its installation directory",
        )

    executable_path = fsldir / "bin" / name
    if not executable_path.is_file():
        raise ApplicationError(
            category="Environment",
            summary=f"FSL {name.upper()} executable not found",
            cause=f"Expected the executable at '{executable_path}'",
            fix="Verify the FSL installation and correct FSLDIR",
            context={"path": str(executable_path)},
        )
    return executable_path


def run_fsl(command: list[str], operation: str) -> None:
    """Run an FSL command and translate process failures."""
    try:
        subprocess.run(command, check=True)
    except OSError as error:
        raise ApplicationError(
            category="Environment",
            summary=f"Could not start FSL {operation}",
            cause=str(error),
            fix=f"Verify that the FSL {operation} executable can run",
            context={"path": command[0]},
        ) from error
    except subprocess.CalledProcessError as error:
        raise ApplicationError(
            category="Preprocessing",
            summary=f"FSL {operation} failed",
            cause=f"FSL {operation} exited with status {error.returncode}",
            fix="Check the input volume and the FSL installation",
            context={"path": command[0]},
        ) from error


def label_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Label connected components in a binary mask."""
    return np.asarray(label(mask, connectivity=connectivity))


def stack_volume_and_frst(volume: np.ndarray, frst: np.ndarray) -> np.ndarray:
    """Return a channel-first model input with volume followed by FRST."""
    return np.stack((volume, frst), axis=0)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original module behind a compiled wrapper when present."""
    original_model = getattr(model, "_orig_mod", None)
    return original_model if isinstance(original_model, nn.Module) else model


def get_model_device(model: nn.Module) -> torch.device:
    """Return the device hosting a model's parameters."""
    return next(model.parameters()).device


def predict_logits(model: nn.Module, model_input: np.ndarray) -> torch.Tensor:
    """Run single-sample inference and return unbatched logits."""
    model.eval()
    model_device = get_model_device(model)
    tensor_input = torch.from_numpy(model_input).float().unsqueeze(0)
    with torch.no_grad():
        return model(tensor_input.to(model_device))[0]


def microbleed_probability(logits: torch.Tensor) -> np.ndarray:
    """Return the microbleed channel of unbatched two-class logits."""
    return F.softmax(logits, dim=0)[1].detach().cpu().numpy()


def initialize_teacher_from_detector(
    detector: CandidateDetector,
    teacher: CandidateDiscriminatorTeacher,
) -> None:
    """Initialize compatible teacher feature and segmentation weights."""
    detector_state = unwrap_model(detector).state_dict()
    teacher_state = unwrap_model(teacher).state_dict()
    transferable = {
        key: value
        for key, value in detector_state.items()
        if key.startswith(("feature_extractor.", "segmentor."))
    }
    missing = [key for key in transferable if key not in teacher_state]
    if missing:
        displayed = ", ".join(missing[:5])
        if len(missing) > 5:  # pragma: no branch
            displayed += f", ... ({len(missing)} total)"
        raise ApplicationError(
            category="Checkpoint",
            summary="Teacher model is incompatible with detector weights",
            cause=f"Teacher is missing detector state keys: {displayed}",
            fix="Use compatible detector and teacher architectures",
        )
    teacher_state.update(transferable)
    unwrap_model(teacher).load_state_dict(teacher_state, strict=True)
