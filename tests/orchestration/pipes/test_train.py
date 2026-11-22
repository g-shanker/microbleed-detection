import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler

from microbleednet.constants import (
    DETECTOR_PATCH_SIZE,
    DETECTOR_STAGE,
    DISCRIMINATOR_PATCH_SIZE,
    STUDENT_STAGE,
    TEACHER_STAGE,
    VALIDATION_AUGMENTATION_FACTOR,
)
from microbleednet.core.common.tasks import (
    KnowledgeDistillationClassificationTask,
    SegmentationClassificationTask,
    SegmentationTask,
)
from microbleednet.core.dataloading.datasets import (
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from microbleednet.core.datamodels import EpochLoss, Hyperparameters, PatchRecord
from microbleednet.errors import ApplicationError
from microbleednet.orchestration.configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from microbleednet.orchestration.layouts import ExperimentLayout, StageName
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PatchManifest,
    TrainManifest,
    TrainStageManifest,
    content_fingerprint,
)
from microbleednet.orchestration.pipes import train
from tests.support import hyperparameters, train_config, train_manifest_for
from tests.support_pipes import (
    ExperimentInputs,
    StubDetector,
    write_experiment_inputs,
)

COMPLETE = ManifestStatus.COMPLETE
RUNNING = ManifestStatus.RUNNING
ALL_STAGES = [DETECTOR_STAGE, TEACHER_STAGE, STUDENT_STAGE]
STAGE_ARGUMENTS = (
    "model",
    "task",
    "train_dataset",
    "validation_dataset",
    "layout",
    "stage",
    "hyperparameters",
    "num_workers",
    "pin_memory",
    "resume",
)
# (stage, patch size, train augmentation factor, dataset, task, recipe field)
STAGE_CASES = [
    pytest.param(
        DETECTOR_STAGE,
        DETECTOR_PATCH_SIZE,
        3,
        SegmentationPatchDataset,
        SegmentationTask,
        "detector_hyperparameters",
        id="detector",
    ),
    pytest.param(
        TEACHER_STAGE,
        DISCRIMINATOR_PATCH_SIZE,
        2,
        SegmentationClassificationPatchDataset,
        SegmentationClassificationTask,
        "teacher_hyperparameters",
        id="teacher",
    ),
    pytest.param(
        STUDENT_STAGE,
        DISCRIMINATOR_PATCH_SIZE,
        2,
        ClassificationPatchDataset,
        KnowledgeDistillationClassificationTask,
        "student_hyperparameters",
        id="student",
    ),
]


def record_in(directory: Path, name: str) -> PatchRecord:
    return PatchRecord(
        volume_path=str(directory / f"{name}-volume.npy"),
        mask_path=str(directory / f"{name}-mask.npy"),
        frst_path=str(directory / f"{name}-frst.npy"),
        patch_index=0,
        has_microbleed=True,
    )


def record_for(stage: str, split: str) -> PatchRecord:
    return record_in(Path(), f"{stage}-{split}")


def write_patch_manifest(
    config: BasePatchConfig,
    records: list[PatchRecord],
    status: ManifestStatus = COMPLETE,
) -> None:
    targeted = isinstance(config, TargetCenteredPatchConfig)
    PatchManifest(
        status=status,
        stage=config.stage,
        split=config.split,
        subject_ids=[subject.subject_id for subject in config.subjects],
        patch_size=config.patch_size,
        augmentation_factor=config.augmentation_factor,
        probability_threshold=config.probability_threshold if targeted else None,
        use_amp=config.use_amp if targeted else False,
        records=records,
    ).write(config.experiment_layout.patch_manifest_path(config.stage, config.split))


def write_stage_manifest(
    layout: ExperimentLayout, stage: StageName, status: ManifestStatus
) -> None:
    TrainStageManifest(
        status=status, stage=stage, hyperparameters=hyperparameters(), history=[]
    ).write(layout.stage_manifest_path(stage))


def fake_extraction(
    monkeypatch: pytest.MonkeyPatch,
    records: list[PatchRecord],
    status: ManifestStatus = COMPLETE,
) -> list[BasePatchConfig]:
    """Replace patch extraction with one that writes ``records`` and logs calls."""
    calls: list[BasePatchConfig] = []

    def execute(config: BasePatchConfig) -> None:
        calls.append(config)
        write_patch_manifest(config, records, status)

    monkeypatch.setattr(train.patch, "execute", execute)
    return calls


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> ExperimentInputs:
    """Read-only dataset and split; every test trains in its own experiment dir."""
    return write_experiment_inputs(
        tmp_path_factory.mktemp("train"), count=4, variants=3
    )


@pytest.fixture(scope="module")
def validated_config(inputs: ExperimentInputs) -> TrainConfig:
    return train_config(
        inputs.root / "dataset",
        inputs.root / "experiment",
        seed=42,
        detector_candidate_threshold=0.7,
        detector_augmentation_factor=3,
        discriminator_augmentation_factor=2,
        num_workers=2,
        pin_memory=True,
        detector_hyperparameters=hyperparameters(batch_size=4, use_amp=True),
        teacher_hyperparameters=hyperparameters(batch_size=6),
        student_hyperparameters=hyperparameters(batch_size=10),
    )


@pytest.fixture
def layout(tmp_path: Path) -> ExperimentLayout:
    return ExperimentLayout(experiment_dir=tmp_path / "experiment")


@pytest.fixture
def config(
    validated_config: TrainConfig, inputs: ExperimentInputs, layout: ExperimentLayout
) -> TrainConfig:
    """The validated config retargeted at a fresh experiment holding the split."""
    split_path = layout.split_manifest_path()
    split_path.parent.mkdir(parents=True)
    shutil.copy(
        ExperimentLayout(
            experiment_dir=validated_config.experiment_dir
        ).split_manifest_path(),
        split_path,
    )
    return validated_config.model_copy(update={"experiment_dir": layout.experiment_dir})


@pytest.fixture
def patch_config(
    layout: ExperimentLayout, inputs: ExperimentInputs
) -> NonOverlappingPatchConfig:
    return NonOverlappingPatchConfig(
        experiment_layout=layout,
        stage=DETECTOR_STAGE,
        split="train",
        subjects=inputs.subjects[:1],
        patch_size=DETECTOR_PATCH_SIZE,
        augmentation_factor=1,
    )


@pytest.fixture
def stage_calls(
    monkeypatch: pytest.MonkeyPatch, layout: ExperimentLayout
) -> list[SimpleNamespace]:
    calls: list[SimpleNamespace] = []

    def record(name: str) -> Callable[..., None]:
        def stage(train_subjects, validation_subjects, stage_layout, device, *deps):
            train_manifest = TrainManifest.read(layout.train_manifest_path())
            calls.append(
                SimpleNamespace(
                    name=name,
                    train=[subject.subject_id for subject in train_subjects],
                    validation=[subject.subject_id for subject in validation_subjects],
                    layout=stage_layout,
                    device=device,
                    deps=deps,
                    manifest_status=train_manifest.status,
                )
            )

        return stage

    for name in ALL_STAGES:
        monkeypatch.setattr(train, f"train_{name}", record(name))
    return calls


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake every heavy collaborator of the stage functions and record their use."""
    state = SimpleNamespace(patch_configs=[], trained=[], loaded=[], initialized=[])

    def extract(config: BasePatchConfig) -> None:
        state.patch_configs.append(config)
        write_patch_manifest(config, [record_for(config.stage, config.split)])

    monkeypatch.setattr(train.patch, "execute", extract)
    monkeypatch.setattr(
        train,
        "train_stage",
        lambda *args: state.trained.append(
            SimpleNamespace(**dict(zip(STAGE_ARGUMENTS, args)))
        ),
    )
    monkeypatch.setattr(train, "CandidateDetector", StubDetector)
    monkeypatch.setattr(train, "CandidateDiscriminatorTeacher", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(train, "CandidateDiscriminatorStudent", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(
        train.utils,
        "load_stage_checkpoint",
        lambda model, path, stage: state.loaded.append((model, path, stage)),
    )
    monkeypatch.setattr(
        train.core_utils,
        "initialize_teacher_from_detector",
        lambda detector, teacher: state.initialized.append((detector, teacher)),
    )
    monkeypatch.setattr(train.utils, "release_gpu_memory", lambda: None)
    return state


@pytest.fixture
def run_stage(
    env: SimpleNamespace,
    config: TrainConfig,
    layout: ExperimentLayout,
    inputs: ExperimentInputs,
) -> Callable[..., None]:
    def run(stage: str, stage_config: TrainConfig = config) -> None:
        getattr(train, f"train_{stage}")(
            inputs.subjects[:2],
            inputs.subjects[2:3],
            layout,
            torch.device("cpu"),
            stage_config,
        )

    return run


def test_execute_runs_stages_in_order_on_split_subjects(
    config: TrainConfig,
    inputs: ExperimentInputs,
    layout: ExperimentLayout,
    stage_calls: list[SimpleNamespace],
) -> None:
    train.execute(config)

    assert [call.name for call in stage_calls] == ALL_STAGES
    for call in stage_calls:
        assert call.train == inputs.split.train_subject_ids
        assert call.validation == inputs.split.validation_subject_ids
        assert call.layout == layout
        assert call.device == torch.device("cpu")
        assert call.deps == (config,)


def test_execute_marks_manifest_running_while_stages_run(
    config: TrainConfig, stage_calls: list[SimpleNamespace]
) -> None:
    train.execute(config)

    assert [call.manifest_status for call in stage_calls] == [RUNNING] * 3


@pytest.mark.parametrize("seed", [42, None])
def test_execute_writes_complete_train_manifest(
    config: TrainConfig,
    inputs: ExperimentInputs,
    stage_calls: list[SimpleNamespace],
    seed: int | None,
) -> None:
    config = config.model_copy(update={"seed": seed})

    train.execute(config)

    manifest = TrainManifest.read(config.experiment_dir / "manifests" / "train.json")
    expected = train_manifest_for(
        config, content_fingerprint(inputs.split), status=COMPLETE
    )
    ignored = {"created_at", "updated_at"}
    assert manifest.model_dump(exclude=ignored) == expected.model_dump(exclude=ignored)


@pytest.mark.parametrize(
    ("resume", "status", "expected"),
    [
        (False, COMPLETE, False),
        (True, None, False),
        (True, RUNNING, False),
        (True, COMPLETE, True),
    ],
)
def test_is_stage_completed(
    layout: ExperimentLayout,
    resume: bool,
    status: ManifestStatus | None,
    expected: bool,
) -> None:
    if status is not None:
        write_stage_manifest(layout, DETECTOR_STAGE, status)

    assert train.is_stage_completed(layout, DETECTOR_STAGE, resume) is expected


def test_extract_patch_records_rejects_incomplete_manifest(
    monkeypatch: pytest.MonkeyPatch, patch_config: NonOverlappingPatchConfig
) -> None:
    fake_extraction(monkeypatch, [], RUNNING)

    with pytest.raises(ApplicationError) as error:
        train.extract_patch_records(patch_config)

    message = str(error.value)
    assert "incomplete patch output" in message
    assert "Rerun the train command with resume set to false" in message


@pytest.mark.parametrize(
    ("resume", "cache", "reused"),
    [
        (True, "valid", True),
        (True, "stale", False),
        (True, None, False),
        (False, "valid", False),
    ],
)
def test_extract_patch_records_reuses_only_valid_cache_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_config: NonOverlappingPatchConfig,
    resume: bool,
    cache: str | None,
    reused: bool,
) -> None:
    cached = record_in(tmp_path, "cached")
    fresh = record_in(tmp_path, "fresh")
    if cache is not None:
        write_patch_manifest(patch_config, [cached])
    if cache == "valid":
        for path in (cached.volume_path, cached.mask_path, cached.frst_path):
            Path(path).touch()
    calls = fake_extraction(monkeypatch, [fresh])

    records = train.extract_patch_records(patch_config, resume=resume)

    assert records == ([cached] if reused else [fresh])
    assert calls == ([] if reused else [patch_config])


@pytest.mark.parametrize(
    ("stage", "patch_size", "train_factor", "_dataset", "_task", "_recipe"),
    STAGE_CASES,
)
def test_stage_extracts_train_and_validation_patches(
    run_stage: Callable[..., None],
    env: SimpleNamespace,
    inputs: ExperimentInputs,
    stage: str,
    patch_size: int,
    train_factor: int,
    _dataset: type,
    _task: type,
    _recipe: str,
) -> None:
    run_stage(stage)

    train_ids = [subject.subject_id for subject in inputs.subjects[:2]]
    validation_ids = [subject.subject_id for subject in inputs.subjects[2:3]]
    assert [
        (
            patch_config.stage,
            patch_config.split,
            patch_config.patch_size,
            patch_config.augmentation_factor,
            [subject.subject_id for subject in patch_config.subjects],
        )
        for patch_config in env.patch_configs
    ] == [
        (stage, "train", patch_size, train_factor, train_ids),
        (
            stage,
            "validation",
            patch_size,
            VALIDATION_AUGMENTATION_FACTOR,
            validation_ids,
        ),
    ]


@pytest.mark.parametrize(
    ("stage", "_size", "_factor", "dataset", "task", "recipe"), STAGE_CASES
)
def test_stage_trains_with_its_recipe_on_extracted_patches(
    run_stage: Callable[..., None],
    env: SimpleNamespace,
    config: TrainConfig,
    stage: str,
    _size: int,
    _factor: int,
    dataset: type,
    task: type,
    recipe: str,
) -> None:
    run_stage(stage, config.model_copy(update={"resume": True}))

    (trained,) = env.trained
    assert trained.stage == stage
    assert isinstance(trained.task, task)
    assert isinstance(trained.train_dataset, dataset)
    assert trained.train_dataset.patches == [record_for(stage, "train")]
    assert isinstance(trained.validation_dataset, dataset)
    assert trained.validation_dataset.patches == [record_for(stage, "validation")]
    assert trained.hyperparameters is getattr(config, recipe)
    assert (trained.num_workers, trained.pin_memory, trained.resume) == (2, True, True)


def test_teacher_is_initialized_from_detector_checkpoint(
    run_stage: Callable[..., None], env: SimpleNamespace, layout: ExperimentLayout
) -> None:
    run_stage(TEACHER_STAGE)

    ((detector, path, stage),) = env.loaded
    ((initializer, teacher),) = env.initialized
    assert isinstance(detector, StubDetector)
    assert detector is initializer
    assert (path, stage) == (
        layout.best_checkpoint_path(DETECTOR_STAGE),
        DETECTOR_STAGE,
    )
    assert env.trained[0].model is teacher


def test_student_extracts_candidates_with_detector_checkpoint(
    run_stage: Callable[..., None],
    env: SimpleNamespace,
    config: TrainConfig,
    layout: ExperimentLayout,
) -> None:
    run_stage(STUDENT_STAGE)

    detector, path, stage = env.loaded[0]
    assert (path, stage) == (
        layout.best_checkpoint_path(DETECTOR_STAGE),
        DETECTOR_STAGE,
    )
    assert len(env.patch_configs) == 2
    for patch_config in env.patch_configs:
        assert isinstance(patch_config, TargetCenteredPatchConfig)
        assert patch_config.detector is detector
        assert patch_config.probability_threshold == config.detector_candidate_threshold
        assert patch_config.use_amp is config.detector_hyperparameters.use_amp


def test_student_distills_from_teacher_checkpoint(
    run_stage: Callable[..., None], env: SimpleNamespace, layout: ExperimentLayout
) -> None:
    run_stage(STUDENT_STAGE)

    teacher, path, stage = env.loaded[1]
    assert (path, stage) == (layout.best_checkpoint_path(TEACHER_STAGE), TEACHER_STAGE)
    assert env.trained[0].task.teacher_model is teacher


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_completed_stage_is_skipped_on_resume(
    run_stage: Callable[..., None],
    env: SimpleNamespace,
    config: TrainConfig,
    layout: ExperimentLayout,
    stage: StageName,
) -> None:
    write_stage_manifest(layout, stage, COMPLETE)

    run_stage(stage, config.model_copy(update={"resume": True}))

    assert (env.patch_configs, env.trained, env.loaded, env.initialized) == (
        [],
        [],
        [],
        [],
    )


@pytest.fixture
def fit(
    monkeypatch: pytest.MonkeyPatch, layout: ExperimentLayout
) -> Callable[..., SimpleNamespace]:
    """Run ``train_stage`` with a fake trainer and loaders; return what it saw."""
    history: list[EpochLoss] = [
        {"epoch": 1, "training_loss": 0.5, "validation_loss": 0.4}
    ]
    datasets = [
        SegmentationPatchDataset([record_for(DETECTOR_STAGE, split)])
        for split in ("train", "validation")
    ]
    state = SimpleNamespace(
        loaders=[],
        resumed=0,
        description=None,
        during_fit=None,
        history=history,
        datasets=datasets,
        recipe=None,
    )

    class FakeTrainer:
        checkpoint_exists = True

        def __init__(
            self, model, task, best_checkpoint, hyperparameters, latest_checkpoint
        ):
            self.latest_checkpoint = latest_checkpoint
            if self.checkpoint_exists:
                latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                latest_checkpoint.write_bytes(b"checkpoint")

        def load_latest_checkpoint(self) -> None:
            state.resumed += 1

        def fit(self, train_loader, validation_loader, description) -> list[EpochLoss]:
            state.description = description
            state.during_fit = TrainStageManifest.read(
                layout.stage_manifest_path(DETECTOR_STAGE)
            )
            return history

    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        train,
        "DataLoader",
        lambda dataset, **kwargs: state.loaders.append((dataset, kwargs)) or object(),
    )
    monkeypatch.setattr(
        train,
        "EqualBatchSampler",
        lambda patches, batch_size: SimpleNamespace(
            patches=patches, batch_size=batch_size
        ),
    )

    def run(
        recipe: Hyperparameters | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        resume: bool = False,
        checkpoint_exists: bool = True,
    ) -> SimpleNamespace:
        FakeTrainer.checkpoint_exists = checkpoint_exists
        state.recipe = recipe or hyperparameters(batch_size=2)
        train.train_stage(
            nn.Linear(1, 1),
            cast(Any, object()),
            datasets[0],
            datasets[1],
            layout,
            DETECTOR_STAGE,
            state.recipe,
            num_workers,
            pin_memory,
            resume,
        )
        return state

    return run


@pytest.mark.parametrize("num_workers", [0, 2])
def test_train_stage_builds_loaders_from_settings(
    fit: Callable[..., SimpleNamespace], num_workers: int
) -> None:
    seen = fit(hyperparameters(batch_size=2), num_workers, pin_memory=True)

    (train_dataset, train_kwargs), (validation_dataset, validation_kwargs) = (
        seen.loaders
    )
    assert (train_dataset, validation_dataset) == tuple(seen.datasets)
    assert train_kwargs["batch_sampler"].patches is seen.datasets[0].patches
    assert train_kwargs["batch_sampler"].batch_size == 2
    assert isinstance(validation_kwargs["batch_sampler"], BatchSampler)
    assert validation_kwargs["batch_sampler"].batch_size == 2
    assert validation_kwargs["batch_sampler"].drop_last is False
    for _, kwargs in seen.loaders:
        assert kwargs["num_workers"] == num_workers
        assert kwargs["pin_memory"] is True
        assert kwargs["persistent_workers"] is (num_workers > 0)


@pytest.mark.parametrize(
    ("resume", "checkpoint_exists", "resumed"),
    [(True, True, 1), (True, False, 0), (False, True, 0)],
)
def test_train_stage_resumes_only_from_existing_checkpoint(
    fit: Callable[..., SimpleNamespace],
    resume: bool,
    checkpoint_exists: bool,
    resumed: int,
) -> None:
    seen = fit(resume=resume, checkpoint_exists=checkpoint_exists)

    assert seen.resumed == resumed


def test_train_stage_fits_under_stage_name(fit: Callable[..., SimpleNamespace]) -> None:
    assert fit().description == DETECTOR_STAGE


def test_train_stage_marks_manifest_running_during_fit(
    fit: Callable[..., SimpleNamespace],
) -> None:
    seen = fit()

    assert seen.during_fit.status is RUNNING
    assert seen.during_fit.hyperparameters == seen.recipe
    assert seen.during_fit.history == []


def test_train_stage_completes_manifest_with_history(
    fit: Callable[..., SimpleNamespace], layout: ExperimentLayout
) -> None:
    seen = fit()

    manifest = TrainStageManifest.read(layout.stage_manifest_path(DETECTOR_STAGE))
    assert manifest.status is COMPLETE
    assert manifest.history == seen.history


def test_train_stage_removes_latest_checkpoint(
    fit: Callable[..., SimpleNamespace], layout: ExperimentLayout
) -> None:
    fit()

    assert not layout.latest_checkpoint_path(DETECTOR_STAGE).exists()
