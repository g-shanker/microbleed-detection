import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label

from ..errors import ApplicationError
from .common.models import CandidateDetector, CandidateDiscriminatorTeacher


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
