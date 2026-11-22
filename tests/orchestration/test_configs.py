from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from microbleednet.constants import DETECTOR_STAGE, STUDENT_STAGE
from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.configs import (
    EvaluateConfig,
    IndexDataConfig,
    InferConfig,
    NonOverlappingPatchConfig,
    PreprocessConfig,
    SplitConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
    ensure_preprocessed_subjects_complete,
    ensure_subjects_have_masks,
    ensure_training_resume_state,
)
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedSubject,
    RawDatasetManifest,
    SplitManifest,
    TrainStageManifest,
    content_fingerprint,
)
from tests.support import (
    hyperparameters,
    make_index_config,
    preprocessed_subjects,
    raw_subjects,
    train_config,
    train_manifest_for,
    write_preprocessed_manifest,
    write_raw_manifest,
    write_split_manifest,
    write_training_dataset,
)

SUBJECT_IDS = [f"subject-{index}" for index in range(7)]
RESUME_HINT = "Set resume to false and rerun preprocessing"
ValuesFactory = Callable[..., dict[str, Any]]


# --- builders -----------------------------------------------------------------


def preprocess_values(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "dataset_dir": tmp_path / "dataset",
        "augmentation_factor": 1,
        **overrides,
    }


def explicit_values(tmp_path: Path, output: str, **overrides: Any) -> dict[str, Any]:
    checkpoints = {}
    for stage in (DETECTOR_STAGE, STUDENT_STAGE):
        checkpoint = tmp_path / f"{stage}.pth"
        checkpoint.touch()
        checkpoints[f"{stage}_checkpoint_path"] = checkpoint
    return {
        "dataset_dir": tmp_path / "dataset",
        "output_dir": tmp_path / output,
        **checkpoints,
        "use_amp": False,
        "device": "cpu",
        **overrides,
    }


def infer_values(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return explicit_values(tmp_path, "inference", **overrides)


def evaluate_values(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return explicit_values(
        tmp_path, "evaluation", **{"bias_field_correction": "fast", **overrides}
    )


def split_config(
    dataset_dir: Path, experiment_dir: Path, **overrides: Any
) -> SplitConfig:
    sizes = {"train_size": 0.7, "validation_size": 0.1, "test_size": 0.2}
    return SplitConfig(
        dataset_dir=dataset_dir, experiment_dir=experiment_dir, **{**sizes, **overrides}
    )


def write_preprocess_checkpoint(
    dataset_dir: Path, *, status: ManifestStatus = ManifestStatus.RUNNING, **overrides
) -> None:
    raw_manifest = RawDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
    )
    overrides.setdefault("raw_manifest_fingerprint", content_fingerprint(raw_manifest))
    write_preprocessed_manifest(dataset_dir, [], status=status, **overrides)


def write_experiment(
    config: TrainConfig,
    layout: ExperimentLayout,
    split_fingerprint: str,
    status: ManifestStatus = ManifestStatus.COMPLETE,
) -> None:
    train_manifest_for(config, split_fingerprint, status=status).write(
        layout.train_manifest_path()
    )
    for stage in (DETECTOR_STAGE, STUDENT_STAGE):
        checkpoint = layout.best_checkpoint_path(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()


def write_detector_stage(
    config: TrainConfig, layout: ExperimentLayout, status: ManifestStatus
) -> None:
    TrainStageManifest(
        status=status,
        stage=DETECTOR_STAGE,
        hyperparameters=config.detector_hyperparameters,
        history=[],
    ).write(layout.stage_manifest_path(DETECTOR_STAGE))


def resumed(config: TrainConfig, **overrides: Any) -> TrainConfig:
    return TrainConfig(**{**config.model_dump(), "resume": True, **overrides})


def without_masks(subject: PreprocessedSubject) -> PreprocessedSubject:
    return subject.model_copy(
        update={
            "variants": [
                variant.model_copy(update={"mask_path": None})
                for variant in subject.variants
            ]
        }
    )


def patch_values(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "experiment_layout": ExperimentLayout(experiment_dir=tmp_path),
        "stage": "student",
        "split": "train",
        "subjects": preprocessed_subjects(1, tmp_path),
        "patch_size": 24,
        "augmentation_factor": 1,
        **overrides,
    }


def assert_incomplete_manifest(
    error: pytest.ExceptionInfo[ValidationError], kind: str, path: Path
) -> None:
    message = str(error.value)
    assert f"Cannot use incomplete {kind} manifest" in message
    assert "Rerun the producing stage to completion" in message
    assert str(path) in message


PREPROCESS = pytest.param(PreprocessConfig, preprocess_values, id="preprocess")
INFER = pytest.param(InferConfig, infer_values, id="infer")
EVALUATE = pytest.param(EvaluateConfig, evaluate_values, id="evaluate")
CONFIG_PARAMS = ("config_cls", "make_values")


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    write_raw_manifest(dataset_dir)
    return dataset_dir


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return write_training_dataset(tmp_path)


@pytest.fixture
def split(tmp_path: Path, dataset_dir: Path) -> SplitManifest:
    return write_split_manifest(tmp_path, SUBJECT_IDS)


@pytest.fixture
def layout(tmp_path: Path) -> ExperimentLayout:
    return ExperimentLayout(experiment_dir=tmp_path / "experiment")


@pytest.fixture
def config(
    dataset_dir: Path, split: SplitManifest, layout: ExperimentLayout
) -> TrainConfig:
    return train_config(dataset_dir, layout.experiment_dir)


# --- IndexDataConfig ----------------------------------------------------------


@pytest.mark.parametrize("source_id", ["siteA", "site-a", "site_a", "123"])
def test_source_id_accepts_filesystem_safe_values(
    tmp_path: Path, source_id: str
) -> None:
    assert make_index_config(tmp_path, source_id=source_id).source_id == source_id


@pytest.mark.parametrize("source_id", ["site/A", "site A", "site@A", ""])
def test_source_id_rejects_unsafe_values(tmp_path: Path, source_id: str) -> None:
    with pytest.raises(ValidationError, match="Source ID is not filesystem-safe"):
        make_index_config(tmp_path, source_id=source_id)


@pytest.mark.parametrize("field", ["source_id", "modality"])
def test_index_config_requires_field(tmp_path: Path, field: str) -> None:
    values = make_index_config(tmp_path).model_dump()
    del values[field]

    with pytest.raises(ValidationError, match=field):
        IndexDataConfig.model_validate(values)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"volume_pattern": "volume.nii.gz"}, "volume_pattern"),
        ({"mask_pattern": "mask.nii.gz"}, "mask_pattern"),
    ],
)
def test_patterns_require_subject_id_placeholder(
    tmp_path: Path, overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(
        ValidationError, match=f"{field} is missing the subject ID placeholder"
    ):
        make_index_config(tmp_path, **overrides)


def test_pattern_accepts_multiple_subject_id_placeholders(tmp_path: Path) -> None:
    pattern = "{subject_id}/{subject_id}_volume.nii.gz"

    assert make_index_config(tmp_path, volume_pattern=pattern).volume_pattern == pattern


def test_mask_inputs_are_optional(tmp_path: Path) -> None:
    values = make_index_config(tmp_path).model_dump()
    del values["mask_dir"], values["mask_pattern"]

    config = IndexDataConfig.model_validate(values)

    assert config.mask_dir is None
    assert config.mask_pattern is None


@pytest.mark.parametrize("omitted", ["mask_dir", "mask_pattern"])
def test_mask_inputs_must_be_provided_together(tmp_path: Path, omitted: str) -> None:
    values = make_index_config(tmp_path).model_dump()
    del values[omitted]

    with pytest.raises(ValidationError, match="must be provided together"):
        IndexDataConfig.model_validate(values)


@pytest.mark.parametrize("field", ["input_dir", "mask_dir"])
def test_index_config_rejects_missing_directories(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValidationError, match=f"Required {field} directory not found"):
        make_index_config(tmp_path, **{field: tmp_path / "missing"})


# --- PreprocessConfig ---------------------------------------------------------


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        (None, "Required dataset_dir directory not found"),
        ("empty", "Required raw manifest file not found"),
        ("unreadable", "Manifest schema validation failed"),
    ],
)
def test_preprocess_config_requires_readable_raw_manifest(
    tmp_path: Path, prepare: str | None, message: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    if prepare:
        manifest_path = DatasetLayout(dataset_dir=dataset_dir).raw_manifest_path()
        manifest_path.parent.mkdir(parents=True)
        if prepare == "unreadable":
            manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        PreprocessConfig.model_validate(preprocess_values(tmp_path))


@pytest.mark.parametrize(CONFIG_PARAMS, [PREPROCESS, INFER, EVALUATE])
def test_raw_manifest_must_be_complete(
    tmp_path: Path, config_cls: type[BaseModel], make_values: ValuesFactory
) -> None:
    write_raw_manifest(tmp_path / "dataset", status=ManifestStatus.RUNNING)
    manifest_path = DatasetLayout(dataset_dir=tmp_path / "dataset").raw_manifest_path()

    with pytest.raises(ValidationError) as error:
        config_cls.model_validate(make_values(tmp_path))

    assert_incomplete_manifest(error, "raw", manifest_path)


@pytest.mark.parametrize(CONFIG_PARAMS, [PREPROCESS, EVALUATE])
def test_bias_field_correction_rejects_unknown_method(
    tmp_path: Path, config_cls: type[BaseModel], make_values: ValuesFactory
) -> None:
    with pytest.raises(ValidationError, match="bias_field_correction"):
        config_cls.model_validate(make_values(tmp_path, bias_field_correction="other"))


def test_preprocess_config_ignores_existing_checkpoint_without_resume(
    tmp_path: Path, raw_dataset: Path
) -> None:
    write_preprocess_checkpoint(raw_dataset, augmentation_factor=99)

    config = PreprocessConfig.model_validate(preprocess_values(tmp_path))

    assert config.resume is False
    assert config.bias_field_correction == "fast"


@pytest.mark.parametrize("checkpoint_seed", [None, 41])
def test_preprocess_resume_accepts_running_checkpoint(
    tmp_path: Path, raw_dataset: Path, checkpoint_seed: int | None
) -> None:
    write_preprocess_checkpoint(raw_dataset, seed=checkpoint_seed)

    config = PreprocessConfig.model_validate(
        preprocess_values(tmp_path, seed=42, resume=True)
    )

    assert (config.resume, config.seed) == (True, 42)


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        (None, "Required preprocessed manifest file not found"),
        ({"status": ManifestStatus.COMPLETE}, "Set resume to false to run the stage"),
        ({"raw_manifest_fingerprint": "stale"}, "fingerprint does not match"),
    ],
)
def test_preprocess_resume_rejects_unusable_checkpoint(
    tmp_path: Path,
    raw_dataset: Path,
    checkpoint: dict[str, Any] | None,
    message: str,
) -> None:
    if checkpoint is not None:
        write_preprocess_checkpoint(raw_dataset, **checkpoint)

    with pytest.raises(ValidationError, match=message):
        PreprocessConfig.model_validate(preprocess_values(tmp_path, resume=True))


# --- resumable preprocessing subjects ----------------------------------------


def test_missing_mask_error_bounds_subject_ids() -> None:
    with pytest.raises(ValueError, match=r"subject-4, \.\.\. \(6 total\)"):
        ensure_subjects_have_masks(raw_subjects(6, with_mask=False))


def test_preprocessed_resume_rejects_unknown_subjects(tmp_path: Path) -> None:
    subjects = preprocessed_subjects(6, tmp_path)

    with pytest.raises(ValueError) as error:
        ensure_preprocessed_subjects_complete(subjects, {"current"}, 1)

    message = str(error.value)
    assert "unknown subjects" in message
    assert "subject-4, ... (6 total)" in message
    assert RESUME_HINT in message


def test_preprocessed_resume_rejects_incomplete_variants(tmp_path: Path) -> None:
    subjects = preprocessed_subjects(1, tmp_path)

    with pytest.raises(ValueError) as error:
        ensure_preprocessed_subjects_complete(subjects, {"subject-0"}, 2)

    message = str(error.value)
    assert "incomplete variants" in message
    assert RESUME_HINT in message


def test_preprocessed_resume_rejects_missing_brain_mask(tmp_path: Path) -> None:
    subject = preprocessed_subjects(1, tmp_path)[0].model_copy(
        update={"brain_mask_path": str(tmp_path / "missing-brain-mask.nii.gz")}
    )

    with pytest.raises(ValueError, match="brain mask is missing"):
        ensure_preprocessed_subjects_complete([subject], {"subject-0"}, 1)


@pytest.mark.parametrize("field", ["volume_path", "frst_path", "mask_path"])
def test_preprocessed_resume_rejects_missing_variant_file(
    tmp_path: Path, field: str
) -> None:
    subject = preprocessed_subjects(1, tmp_path, variants=2)[0]
    missing = Path(getattr(subject.variants[1], field))
    missing.unlink()

    with pytest.raises(ValueError) as error:
        ensure_preprocessed_subjects_complete([subject], {"subject-0"}, 2)

    message = str(error.value)
    assert "output is missing" in message
    assert f"variant 1 references missing files: {missing}" in message
    assert RESUME_HINT in message


@pytest.mark.parametrize("with_masks", [True, False])
def test_preprocessed_resume_accepts_existing_files(
    tmp_path: Path, with_masks: bool
) -> None:
    subjects = preprocessed_subjects(2, tmp_path, variants=2, with_masks=with_masks)

    ensure_preprocessed_subjects_complete(
        subjects, {subject.subject_id for subject in subjects}, 2
    )


# --- SplitConfig and TrainConfig inputs --------------------------------------


@pytest.mark.parametrize("field", ["train_size", "validation_size", "test_size"])
def test_split_config_requires_each_size(tmp_path: Path, field: str) -> None:
    values = {
        "dataset_dir": tmp_path,
        "experiment_dir": tmp_path,
        "train_size": 0.7,
        "validation_size": 0.1,
        "test_size": 0.2,
    }
    del values[field]

    with pytest.raises(ValidationError, match=field):
        SplitConfig.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train_size", 0.0, "greater than 0"),
        ("validation_size", 1.0, "less than 1"),
        ("test_size", 0.0, "greater than 0"),
    ],
)
def test_split_config_rejects_out_of_range_size(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    with pytest.raises(ValidationError, match=f"(?s){field}.*{message}"):
        split_config(tmp_path, tmp_path, **{field: value})


def test_split_config_requires_sizes_to_sum_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Split proportions must total 1.0"):
        split_config(tmp_path, tmp_path, train_size=0.6, validation_size=0.1)


def test_split_config_accepts_complete_dataset(
    tmp_path: Path, dataset_dir: Path
) -> None:
    config = split_config(dataset_dir, tmp_path / "experiment", seed=3)

    assert (config.train_size, config.seed) == (0.7, 3)


def test_split_config_rejects_subjects_without_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    subjects = preprocessed_subjects(2, tmp_path / "inputs", with_masks=False)
    write_preprocessed_manifest(dataset_dir, subjects)

    with pytest.raises(ValidationError, match="Required subject masks are missing"):
        split_config(dataset_dir, tmp_path / "experiment")


def _missing_dataset(tmp_path: Path) -> Path:
    return tmp_path / "missing"


def _empty_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    return dataset_dir


def _running_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "running"
    write_preprocessed_manifest(dataset_dir, [], status=ManifestStatus.RUNNING)
    return dataset_dir


@pytest.mark.parametrize("build", [split_config, train_config])
@pytest.mark.parametrize(
    ("make_dataset", "message"),
    [
        (_missing_dataset, "Required dataset_dir directory not found"),
        (_empty_dataset, "Required preprocessed manifest file not found"),
        (_running_dataset, "Cannot use incomplete"),
    ],
)
def test_dataset_inputs_are_validated(
    tmp_path: Path,
    build: Callable[..., Any],
    make_dataset: Callable[[Path], Path],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        build(make_dataset(tmp_path), tmp_path / "experiment")


# --- TrainConfig --------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "message"),
    [
        ("invalid", "Invalid compute device"),
        ("cuda", "Requested CUDA device is unavailable"),
    ],
)
def test_train_config_rejects_unusable_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, message: str
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(ValidationError, match=message):
        train_config(tmp_path, tmp_path, device=device)


def test_train_config_requires_training_recipe(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as error:
        TrainConfig.model_validate(
            {"dataset_dir": tmp_path, "experiment_dir": tmp_path}
        )

    for field in (
        "device",
        "detector_candidate_threshold",
        "detector_augmentation_factor",
        "discriminator_augmentation_factor",
        "num_workers",
        "pin_memory",
        "detector_hyperparameters",
        "teacher_hyperparameters",
        "student_hyperparameters",
    ):
        assert field in str(error.value)


@pytest.mark.parametrize(
    "field",
    [
        "detector_candidate_threshold",
        "detector_augmentation_factor",
        "discriminator_augmentation_factor",
        "num_workers",
    ],
)
def test_train_config_rejects_out_of_range_values(tmp_path: Path, field: str) -> None:
    invalid = {"detector_candidate_threshold": 1.1, "num_workers": -1}

    with pytest.raises(ValidationError, match=field):
        train_config(tmp_path, tmp_path, **{field: invalid.get(field, 0)})


def test_train_config_accepts_stage_training_settings(
    dataset_dir: Path, split: SplitManifest, layout: ExperimentLayout
) -> None:
    config = train_config(
        dataset_dir,
        layout.experiment_dir,
        pin_memory=True,
        detector_hyperparameters=hyperparameters(batch_size=4, max_epochs=12),
        teacher_hyperparameters=hyperparameters(batch_size=6, max_epochs=13),
        student_hyperparameters=hyperparameters(batch_size=6, max_epochs=14),
    )

    assert config.pin_memory is True
    assert [
        (recipe.batch_size, recipe.max_epochs)
        for recipe in (
            config.detector_hyperparameters,
            config.teacher_hyperparameters,
            config.student_hyperparameters,
        )
    ] == [(4, 12), (6, 13), (6, 14)]


def test_hyperparameters_require_even_batch_size() -> None:
    with pytest.raises(ValidationError, match="multiple of 2"):
        hyperparameters(batch_size=3)


@pytest.mark.parametrize("factor_name", ["detector", "discriminator"])
def test_train_config_rejects_unavailable_augmentation_factor(
    dataset_dir: Path, layout: ExperimentLayout, factor_name: str
) -> None:
    with pytest.raises(
        ValidationError, match=f"augmentation factor exceeds.*{factor_name}=2"
    ):
        train_config(
            dataset_dir,
            layout.experiment_dir,
            **{f"{factor_name}_augmentation_factor": 2},
        )


def test_train_config_requires_split_manifest(
    dataset_dir: Path, layout: ExperimentLayout
) -> None:
    with pytest.raises(ValidationError, match="Required split manifest file not found"):
        train_config(dataset_dir, layout.experiment_dir)


def test_train_config_rejects_running_split_manifest(
    tmp_path: Path, dataset_dir: Path, layout: ExperimentLayout
) -> None:
    write_split_manifest(tmp_path, SUBJECT_IDS, status=ManifestStatus.RUNNING)

    with pytest.raises(ValidationError) as error:
        train_config(dataset_dir, layout.experiment_dir)

    assert_incomplete_manifest(error, "split", layout.split_manifest_path())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_dir": "other-dataset"}, "belongs to a different dataset"),
        ({"preprocessed_manifest_fingerprint": "stale"}, "fingerprint does not match"),
    ],
)
def test_train_config_rejects_inconsistent_split_manifest(
    tmp_path: Path,
    dataset_dir: Path,
    layout: ExperimentLayout,
    overrides: dict[str, Any],
    message: str,
) -> None:
    write_split_manifest(tmp_path, SUBJECT_IDS, **overrides)

    with pytest.raises(ValidationError, match=message):
        train_config(dataset_dir, layout.experiment_dir)


# --- TrainConfig resume -------------------------------------------------------


def test_resume_requires_train_manifest(config: TrainConfig) -> None:
    with pytest.raises(ValidationError, match="Required train manifest file not found"):
        resumed(config)


def test_resume_rejects_completed_train_manifest(
    config: TrainConfig, layout: ExperimentLayout, split: SplitManifest
) -> None:
    write_experiment(config, layout, content_fingerprint(split))

    with pytest.raises(ValidationError, match="Cannot resume a completed manifest"):
        resumed(config)


@pytest.mark.parametrize("stage_status", [None, ManifestStatus.RUNNING])
def test_resume_requires_recoverable_training_state(
    config: TrainConfig,
    layout: ExperimentLayout,
    split: SplitManifest,
    stage_status: ManifestStatus | None,
) -> None:
    write_experiment(config, layout, content_fingerprint(split), ManifestStatus.RUNNING)
    if stage_status is not None:
        write_detector_stage(config, layout, stage_status)

    with pytest.raises(ValidationError, match="No recoverable training state found"):
        resumed(config)


@pytest.mark.parametrize(
    "changes",
    [{}, {"detector_hyperparameters": hyperparameters(max_epochs=101)}],
    ids=["unchanged", "changed-settings"],
)
def test_resume_accepts_completed_stage(
    config: TrainConfig,
    layout: ExperimentLayout,
    split: SplitManifest,
    changes: dict[str, Any],
) -> None:
    write_experiment(config, layout, content_fingerprint(split), ManifestStatus.RUNNING)
    write_detector_stage(config, layout, ManifestStatus.COMPLETE)

    assert resumed(config, **changes).resume is True


def test_resume_rejects_changed_split(
    config: TrainConfig, layout: ExperimentLayout, split: SplitManifest
) -> None:
    write_experiment(config, layout, content_fingerprint(split), ManifestStatus.RUNNING)
    write_detector_stage(config, layout, ManifestStatus.COMPLETE)
    split.model_copy(update={"seed": 99}).write(layout.split_manifest_path())

    with pytest.raises(ValidationError, match="fingerprint does not match"):
        resumed(config)


def test_training_resume_state_accepts_latest_checkpoint(
    layout: ExperimentLayout,
) -> None:
    checkpoint = layout.latest_checkpoint_path(DETECTOR_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    ensure_training_resume_state(layout)


# --- InferConfig and explicit EvaluateConfig ---------------------------------


@pytest.mark.parametrize(
    ("config_cls", "make_values", "field"),
    [
        (InferConfig, infer_values, field)
        for field in (
            "dataset_dir",
            "output_dir",
            "detector_checkpoint_path",
            "student_checkpoint_path",
            "use_amp",
            "device",
        )
    ]
    + [
        (EvaluateConfig, evaluate_values, field)
        for field in (
            "dataset_dir",
            "output_dir",
            "detector_checkpoint_path",
            "student_checkpoint_path",
            "bias_field_correction",
            "use_amp",
            "device",
        )
    ],
)
def test_explicit_config_requires_field(
    tmp_path: Path,
    config_cls: type[BaseModel],
    make_values: ValuesFactory,
    field: str,
) -> None:
    values = make_values(tmp_path)
    del values[field]

    with pytest.raises(ValidationError, match=field):
        config_cls.model_validate(values)


@pytest.mark.parametrize("stage", [DETECTOR_STAGE, STUDENT_STAGE])
@pytest.mark.parametrize(CONFIG_PARAMS, [INFER, EVALUATE])
def test_explicit_config_requires_existing_checkpoints(
    tmp_path: Path,
    raw_dataset: Path,
    config_cls: type[BaseModel],
    make_values: ValuesFactory,
    stage: str,
) -> None:
    values = make_values(tmp_path, **{f"{stage}_checkpoint_path": tmp_path / "gone"})

    with pytest.raises(ValidationError, match=f"Required {stage} checkpoint file"):
        config_cls.model_validate(values)


def test_infer_config_accepts_raw_dataset(tmp_path: Path, raw_dataset: Path) -> None:
    config = InferConfig.model_validate(infer_values(tmp_path))

    assert config.dataset_dir == raw_dataset
    assert config.model_dump(mode="json")["dataset_dir"] == str(raw_dataset)


def test_evaluate_config_accepts_explicit_checkpoints(
    tmp_path: Path, raw_dataset: Path
) -> None:
    config = EvaluateConfig.model_validate(evaluate_values(tmp_path))

    assert config.output_dir == tmp_path / "evaluation"
    assert config.bias_field_correction == "fast"


def test_evaluate_config_rejects_maskless_raw_subjects(tmp_path: Path) -> None:
    write_raw_manifest(tmp_path / "dataset", raw_subjects(with_mask=False))

    with pytest.raises(ValidationError, match="Required subject masks are missing"):
        EvaluateConfig.model_validate(evaluate_values(tmp_path))


# --- EvaluateConfig from an experiment ---------------------------------------


def experiment_evaluate_values(
    dataset_dir: Path, experiment_dir: Path
) -> dict[str, Any]:
    return {
        "dataset_dir": dataset_dir,
        "experiment_dir": experiment_dir,
        "bias_field_correction": "fast",
        "use_amp": False,
        "device": "cpu",
    }


def test_evaluate_config_derives_paths_from_experiment(
    config: TrainConfig, layout: ExperimentLayout, split: SplitManifest
) -> None:
    write_experiment(config, layout, content_fingerprint(split))

    evaluated = EvaluateConfig.model_validate(
        experiment_evaluate_values(config.dataset_dir, layout.experiment_dir)
    )

    assert evaluated.output_dir == layout.experiment_dir
    assert evaluated.detector_checkpoint_path == layout.best_checkpoint_path(
        DETECTOR_STAGE
    )
    assert evaluated.student_checkpoint_path == layout.best_checkpoint_path(
        STUDENT_STAGE
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"status": ManifestStatus.RUNNING}, "Cannot use incomplete"),
        ({"split_fingerprint": "stale"}, "fingerprint does not match"),
    ],
)
def test_evaluate_config_rejects_inconsistent_train_manifest(
    config: TrainConfig,
    layout: ExperimentLayout,
    split: SplitManifest,
    state: dict[str, Any],
    message: str,
) -> None:
    state = {"split_fingerprint": content_fingerprint(split), **state}
    write_experiment(config, layout, **state)

    with pytest.raises(ValidationError, match=message):
        EvaluateConfig.model_validate(
            experiment_evaluate_values(config.dataset_dir, layout.experiment_dir)
        )


def test_evaluate_config_rejects_changed_preprocessed_manifest(
    config: TrainConfig, layout: ExperimentLayout, split: SplitManifest
) -> None:
    write_experiment(config, layout, content_fingerprint(split))
    write_preprocessed_manifest(config.dataset_dir, [])

    with pytest.raises(ValidationError, match="fingerprint does not match"):
        EvaluateConfig.model_validate(
            experiment_evaluate_values(config.dataset_dir, layout.experiment_dir)
        )


def test_evaluate_config_rejects_maskless_test_subject(
    tmp_path: Path, layout: ExperimentLayout
) -> None:
    dataset_dir = tmp_path / "dataset"
    train_subject, test_subject = preprocessed_subjects(2, tmp_path / "inputs")
    write_preprocessed_manifest(
        dataset_dir, [train_subject, without_masks(test_subject)]
    )
    split = write_split_manifest(tmp_path, ["subject-0"], test_ids=["subject-1"])
    config = train_config(dataset_dir, layout.experiment_dir)
    write_experiment(config, layout, content_fingerprint(split))

    with pytest.raises(ValidationError, match="Required subject masks are missing"):
        EvaluateConfig.model_validate(
            experiment_evaluate_values(dataset_dir, layout.experiment_dir)
        )


# --- Patch configs ------------------------------------------------------------


def test_patch_config_rejects_unavailable_augmentation_factor(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Patch augmentation factor exceeds"):
        NonOverlappingPatchConfig.model_validate(
            patch_values(tmp_path, augmentation_factor=2)
        )


def test_patch_config_rejects_variant_without_mask(tmp_path: Path) -> None:
    subjects = preprocessed_subjects(1, tmp_path, with_masks=False)

    with pytest.raises(ValidationError, match="requires a variant mask"):
        NonOverlappingPatchConfig.model_validate(
            patch_values(tmp_path, subjects=subjects)
        )


@pytest.mark.parametrize(
    ("field", "name"),
    [("volume_path", "volume"), ("mask_path", "mask"), ("frst_path", "FRST")],
)
def test_patch_config_rejects_missing_input_file(
    tmp_path: Path, field: str, name: str
) -> None:
    values = patch_values(tmp_path)
    Path(getattr(values["subjects"][0].variants[0], field)).unlink()

    with pytest.raises(ValidationError, match=f"variant 0 {name} file was not found"):
        NonOverlappingPatchConfig.model_validate(values)


def test_target_centered_config_accepts_candidate_detector(tmp_path: Path) -> None:
    config = TargetCenteredPatchConfig.model_validate(
        patch_values(
            tmp_path,
            probability_threshold=0.5,
            use_amp=False,
            detector=CandidateDetector(),
        )
    )

    assert config.probability_threshold == 0.5


def test_target_centered_config_rejects_invalid_detector(tmp_path: Path) -> None:
    values = patch_values(
        tmp_path, probability_threshold=0.5, use_amp=False, detector=object()
    )

    with pytest.raises(TypeError, match="Target-centered patch extraction"):
        TargetCenteredPatchConfig.model_validate(values)
