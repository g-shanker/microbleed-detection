from pathlib import Path

import numpy as np
import pytest
import torch

from microbleednet.core.dataloading.datasets import (
    BasePatchDataset,
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from microbleednet.core.datamodels import PatchRecord


def _record(tmp_path: Path) -> PatchRecord:
    volume_path = tmp_path / "volumes.npy"
    mask_path = tmp_path / "masks.npy"
    frst_path = tmp_path / "frst.npy"
    np.save(volume_path, np.ones((1, 4, 4, 4)))
    np.save(mask_path, np.ones((1, 4, 4, 4), dtype=np.uint8))
    np.save(frst_path, np.full((1, 4, 4, 4), 2.0))
    return PatchRecord(
        volume_path=str(volume_path),
        mask_path=str(mask_path),
        patch_index=0,
        has_microbleed=True,
        frst_path=str(frst_path),
    )


def test_patch_datasets_return_expected_batches(tmp_path: Path) -> None:
    records = [_record(tmp_path)]

    segmentation = SegmentationPatchDataset(records)[0]
    combined = SegmentationClassificationPatchDataset(records)[0]
    classification = ClassificationPatchDataset(records)[0]

    assert segmentation.volume.shape == (2, 4, 4, 4)
    assert segmentation.mask.dtype == torch.long
    assert combined.label.item() == 1
    assert classification.label.item() == 1


def test_dataset_reuses_mmap_and_loads_persisted_frst(tmp_path: Path) -> None:
    record = _record(tmp_path)
    dataset = SegmentationPatchDataset([record])

    dataset[0]
    dataset[0]

    assert len(dataset) == 1
    assert len(dataset.mmaps) == 3
    assert SegmentationClassificationPatchDataset([record])[0].label.item() == 1
    assert ClassificationPatchDataset([record])[0].label.item() == 1
    assert torch.all(dataset[0].volume[1] == 2)


def test_base_dataset_requires_getitem() -> None:
    with pytest.raises(NotImplementedError):
        BasePatchDataset([])[0]