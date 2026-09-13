import pytest

from microbleednet.core.dataloading.samplers import EqualBatchSampler
from microbleednet.core.datamodels import PatchRecord


def _patch(positive: bool) -> PatchRecord:
    return PatchRecord(
        volume_path="volume",
        mask_path="mask",
        frst_path="frst",
        patch_index=0,
        has_microbleed=positive,
        augmented=False,
    )


def test_equal_batch_sampler_is_balanced() -> None:
    patches = [_patch(True), _patch(False)]
    sampler = EqualBatchSampler(patches, batch_size=4)
    batches = list(sampler)

    assert len(batches) == len(sampler) == 1
    assert sum(patches[index].has_microbleed for index in batches[0]) == 2


@pytest.mark.parametrize("batch_size", [0, 3])
def test_equal_batch_sampler_requires_positive_even_batch_size(
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive even"):
        EqualBatchSampler([_patch(True), _patch(False)], batch_size)


def test_equal_batch_sampler_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        EqualBatchSampler([_patch(True)], batch_size=2)