from microbleednet.core.datamodels import PatchSizes


def test_patch_sizes_are_fixed_model_input_contracts() -> None:
    assert PatchSizes.DETECTOR == 48
    assert PatchSizes.DISCRIMINATOR == 24
