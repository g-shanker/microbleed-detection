from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.models import CandidateDetector, CandidateDiscriminatorTeacher
from .transforms import frst

AMP_DTYPE = torch.float16


def unwrap_model(model: nn.Module) -> nn.Module:
    return cast(nn.Module, model._orig_mod if hasattr(model, "_orig_mod") else model)


def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def append_frst_channel(volume: torch.Tensor) -> torch.Tensor:
    """Return ``volume`` with its FRST transform appended as a second channel."""
    return torch.cat((volume, frst.apply(volume)), dim=1)


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
