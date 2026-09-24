from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..datamodels import LoadedPatch, PatchRecord
from ..io import load_array_mmap
from ..utils import stack_volume_and_frst


class SegmentationBatch(NamedTuple):
    volume: torch.Tensor
    mask: torch.Tensor


class SegmentationClassificationBatch(NamedTuple):
    volume: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor


class ClassificationBatch(NamedTuple):
    volume: torch.Tensor
    label: torch.Tensor


class BasePatchDataset(Dataset):
    def __init__(
        self,
        patches: list[PatchRecord],
    ):
        self.patches = patches
        self.mmaps: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.patches)

    def mmap(self, path: str) -> np.ndarray:
        array = self.mmaps.get(path)
        if array is None:
            array = load_array_mmap(path)
            self.mmaps[path] = array
        return array

    def load_patch(self, idx: int) -> LoadedPatch:
        record = self.patches[idx]
        volume = np.array(self.mmap(record.volume_path)[record.patch_index])
        mask = np.array(self.mmap(record.mask_path)[record.patch_index])
        frst = np.array(self.mmap(record.frst_path)[record.patch_index])

        volume = stack_volume_and_frst(volume, frst)

        return LoadedPatch(
            volume=volume,
            mask=mask,
            has_microbleed=record.has_microbleed,
        )

    def __getitem__(self, idx: int):
        raise NotImplementedError("BasePatchDataset.__getitem__ must be implemented")


class SegmentationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int) -> SegmentationBatch:
        patch = self.load_patch(idx)

        volume = patch.volume
        mask = patch.mask

        return SegmentationBatch(
            volume=torch.from_numpy(volume).float(),
            mask=torch.from_numpy(mask).long(),
        )


class SegmentationClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int) -> SegmentationClassificationBatch:
        patch = self.load_patch(idx)

        volume = patch.volume
        mask = patch.mask
        label = patch.has_microbleed
        return SegmentationClassificationBatch(
            volume=torch.from_numpy(volume).float(),
            mask=torch.from_numpy(mask).long(),
            label=torch.tensor(label).long(),
        )


class ClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int) -> ClassificationBatch:
        patch = self.load_patch(idx)

        volume = patch.volume
        label = patch.has_microbleed
        return ClassificationBatch(
            volume=torch.from_numpy(volume).float(),
            label=torch.tensor(label).long(),
        )
