from pathlib import Path

import numpy as np
import pytest
import torch

from microbleednet.core.dataloading import datasets
from microbleednet.core.dataloading.datasets import (
    BasePatchDataset,
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from tests.support_cli_core import patch_record

SEGMENTATION_DATASETS = [
    SegmentationPatchDataset,
    SegmentationClassificationPatchDataset,
]
LABELLED_DATASETS = [SegmentationClassificationPatchDataset, ClassificationPatchDataset]
ALL_DATASETS = [SegmentationPatchDataset, *LABELLED_DATASETS]


@pytest.mark.parametrize("dataset_class", ALL_DATASETS)
def test_item_volume_stacks_volume_and_persisted_frst(
    tmp_path: Path, dataset_class: type[BasePatchDataset]
) -> None:
    volume = dataset_class([patch_record(tmp_path)])[0].volume

    assert volume.shape == (2, 4, 4, 4)
    assert volume.dtype == torch.float32
    assert torch.all(volume[0] == 1)
    assert torch.all(volume[1] == 2)


@pytest.mark.parametrize("dataset_class", SEGMENTATION_DATASETS)
def test_item_mask_is_long(
    tmp_path: Path, dataset_class: type[BasePatchDataset]
) -> None:
    mask = dataset_class([patch_record(tmp_path)])[0].mask

    assert mask.dtype == torch.long
    assert torch.all(mask == 1)


@pytest.mark.parametrize("dataset_class", LABELLED_DATASETS)
@pytest.mark.parametrize("positive", [True, False])
def test_item_label_reflects_microbleed_presence(
    tmp_path: Path, dataset_class: type[BasePatchDataset], positive: bool
) -> None:
    label = dataset_class([patch_record(tmp_path, positive=positive)])[0].label

    assert label.dtype == torch.long
    assert label.item() == int(positive)


def test_dataset_length_counts_patch_records(tmp_path: Path) -> None:
    records = [patch_record(tmp_path), patch_record(tmp_path, positive=False)]

    assert len(SegmentationPatchDataset(records)) == 2


def test_dataset_reuses_open_mmaps_across_items(tmp_path: Path) -> None:
    dataset = SegmentationPatchDataset([patch_record(tmp_path)])

    dataset[0]
    first = dict(dataset.mmaps)
    dataset[0]

    assert len(first) == 3
    assert all(dataset.mmaps[path] is mmap for path, mmap in first.items())


def test_dataset_evicts_least_recently_used_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "MAX_OPEN_PATCH_FILES", 2)
    paths = [str(tmp_path / f"array-{index}.npy") for index in range(3)]
    for index, path in enumerate(paths):
        np.save(path, np.array([index]))
    dataset = BasePatchDataset([])

    for index in (0, 1, 0, 2):
        dataset.mmap(paths[index])

    assert list(dataset.mmaps) == [paths[0], paths[2]]


def test_base_dataset_requires_getitem() -> None:
    with pytest.raises(NotImplementedError):
        BasePatchDataset([])[0]
