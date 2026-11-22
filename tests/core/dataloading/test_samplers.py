import pytest

from microbleednet.core.dataloading.samplers import EqualBatchSampler
from microbleednet.errors import ApplicationError
from tests.support_cli_core import patch_record


@pytest.mark.parametrize(
    ("positives", "negatives", "batch_size", "expected_batches"),
    [(1, 1, 4, 1), (3, 1, 2, 3), (1, 5, 4, 3)],
)
def test_equal_batch_sampler_yields_balanced_batches(
    positives: int, negatives: int, batch_size: int, expected_batches: int
) -> None:
    patches = [patch_record(positive=True)] * positives + [
        patch_record(positive=False)
    ] * negatives
    sampler = EqualBatchSampler(patches, batch_size=batch_size)

    batches = list(sampler)

    assert len(batches) == len(sampler) == expected_batches
    for batch in batches:
        assert len(batch) == batch_size
        assert sum(patches[index].has_microbleed for index in batch) == batch_size // 2


@pytest.mark.parametrize(
    ("positive", "missing"), [(True, "negative"), (False, "positive")]
)
def test_equal_batch_sampler_requires_both_classes(
    positive: bool, missing: str
) -> None:
    with pytest.raises(ApplicationError, match=f"no {missing} examples"):
        EqualBatchSampler([patch_record(positive=positive)], batch_size=2)
