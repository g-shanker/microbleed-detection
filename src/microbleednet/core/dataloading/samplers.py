import torch
from torch.utils.data import Sampler

from ...errors import ApplicationError
from ..datamodels import PatchRecord


class EqualBatchSampler(Sampler):
    def __init__(
        self,
        patches: list[PatchRecord],
        batch_size: int,
    ):
        self.batch_size = batch_size

        self.pos_indices = [
            i for i, patch in enumerate(patches) if patch.has_microbleed
        ]
        self.neg_indices = [
            i for i, patch in enumerate(patches) if not patch.has_microbleed
        ]

        if not self.pos_indices or not self.neg_indices:
            missing_class = "positive" if not self.pos_indices else "negative"
            raise ApplicationError(
                category="Input data",
                summary=f"Training patches contain no {missing_class} examples",
                fix=(
                    "Adjust patch extraction or data composition to include both "
                    "classes"
                ),
            )

        self.num_pos = self.batch_size // 2
        self.num_neg = self.batch_size // 2
        self.num_batches = max(
            (len(self.pos_indices) + self.num_pos - 1) // self.num_pos,
            (len(self.neg_indices) + self.num_neg - 1) // self.num_neg,
        )

    def __iter__(self):
        for _ in range(self.num_batches):
            positive = torch.randint(
                len(self.pos_indices), (self.num_pos,)
            )
            negative = torch.randint(
                len(self.neg_indices), (self.num_neg,)
            )
            batch = [self.pos_indices[index] for index in positive.tolist()]
            batch.extend([self.neg_indices[index] for index in negative.tolist()])
            order = torch.randperm(self.batch_size).tolist()
            yield [batch[index] for index in order]

    def __len__(self):
        return self.num_batches
