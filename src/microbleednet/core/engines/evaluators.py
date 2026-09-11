import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from .. import utils
from ..common.tasks import BaseTask


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        use_amp: bool,
    ):
        self.model = model
        self.device = utils.get_model_device(model)
        self.task = task
        self.use_amp = use_amp

    def validation_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if self.use_amp:
                    with autocast(
                        device_type=self.device.type,
                        dtype=utils.AMP_DTYPE,
                    ):
                        loss = self.task.validation_step(self.model, batch)
                else:
                    loss = self.task.validation_step(self.model, batch)
                batch_size = batch.volume.shape[0]
                running_loss += loss.item() * batch_size
                sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot calculate validation loss for an empty DataLoader")

        return running_loss / sample_count
