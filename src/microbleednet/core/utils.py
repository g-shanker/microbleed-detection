from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label

from .common.models import CandidateDetector, CandidateDiscriminatorTeacher

AMP_DTYPE = torch.float16
COMPONENT_CONNECTIVITY = 3


def label_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Label connected components in a binary mask."""
    return cast(np.ndarray, label(mask, connectivity=connectivity))


def stack_volume_and_frst(volume: np.ndarray, frst: np.ndarray) -> np.ndarray:
    """Return a channel-first model input with volume followed by FRST."""
    return np.stack((volume, frst), axis=0)


def unwrap_model(model: nn.Module) -> nn.Module:
    return cast(nn.Module, model._orig_mod if hasattr(model, "_orig_mod") else model)


def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def predict_logits(model: nn.Module, model_input: np.ndarray) -> torch.Tensor:
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
    detector_state = unwrap_model(detector).state_dict()
    teacher_state = unwrap_model(teacher).state_dict()
    transferable = {
        key: value
        for key, value in detector_state.items()
        if key.startswith(("feature_extractor.", "segmentor."))
    }
    missing = [key for key in transferable if key not in teacher_state]
    if missing:
        raise RuntimeError(f"detector-to-teacher keys missing in teacher: {missing}")
    teacher_state.update(transferable)
    unwrap_model(teacher).load_state_dict(teacher_state, strict=True)
