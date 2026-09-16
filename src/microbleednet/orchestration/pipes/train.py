import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from ...core import io as core_io
from ...core import utils as core_utils
from ...core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
)
from ...core.common.tasks import (
    BaseTask,
    KnowledgeDistillationClassificationTask,
    SegmentationClassificationTask,
    SegmentationTask,
)
from ...core.dataloading.datasets import (
    BasePatchDataset,
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from ...core.dataloading.samplers import EqualBatchSampler
from ...core.datamodels import (
    EpochLoss,
    Hyperparameters,
    PatchRecord,
    PatchSizes,
)
from ...core.engines.trainers import Trainer
from .. import utils
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from ..layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    TEACHER_STAGE,
    DatasetLayout,
    ExperimentLayout,
    StageName,
)
from ..manifests import (
    ManifestStatus,
    PatchManifest,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    SplitManifest,
    TrainManifest,
)
from . import patch

VALIDATION_AUGMENTATION_FACTOR = 1


def execute(config: TrainConfig) -> None:
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    experiment_layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        dataset_layout.preprocessed_manifest_path()
    )
    split_manifest = SplitManifest.read(experiment_layout.split_manifest_path())
    train_subjects = utils.resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.train_subject_ids
    )
    validation_subjects = utils.resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.validation_subject_ids
    )
    device = torch.device(config.device)
    detector_history = train_detector(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    teacher_history = train_teacher(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    student_history = train_student(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    write_train_manifest(
        experiment_layout,
        config,
        detector_history,
        teacher_history,
        student_history,
    )


def write_train_manifest(
    experiment_layout: ExperimentLayout,
    config: TrainConfig,
    detector_history: list[EpochLoss],
    teacher_history: list[EpochLoss],
    student_history: list[EpochLoss],
) -> None:
    TrainManifest(
        status=ManifestStatus.COMPLETE,
        dataset_dir=str(config.dataset_dir.resolve()),
        device=config.device,
        seed=config.seed,
        detector_candidate_threshold=config.detector_candidate_threshold,
        detector_augmentation_factor=config.detector_augmentation_factor,
        discriminator_augmentation_factor=config.discriminator_augmentation_factor,
        validation_augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        detector_patch_size=PatchSizes.DETECTOR,
        discriminator_patch_size=PatchSizes.DISCRIMINATOR,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        detector_hyperparameters=config.detector_hyperparameters,
        teacher_hyperparameters=config.teacher_hyperparameters,
        student_hyperparameters=config.student_hyperparameters,
        detector_history=detector_history,
        teacher_history=teacher_history,
        student_history=student_history,
    ).write(experiment_layout.train_manifest_path())


def train_stage(
    model: nn.Module,
    task: BaseTask,
    train_dataset: BasePatchDataset,
    validation_dataset: BasePatchDataset,
    experiment_layout: ExperimentLayout,
    stage: StageName,
    hyperparameters: Hyperparameters,
    num_workers: int,
    pin_memory: bool,
) -> list[EpochLoss]:
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=EqualBatchSampler(
            train_dataset.patches,
            batch_size=hyperparameters.batch_size,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(validation_dataset),
            batch_size=hyperparameters.batch_size,
            drop_last=False,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return Trainer(
        model,
        task,
        best_checkpoint=experiment_layout.best_checkpoint_path(stage),
        hyperparameters=hyperparameters,
    ).fit(train_loader, validation_loader)


def extract_patch_records(config: BasePatchConfig) -> list[PatchRecord]:
    patch.execute(config)
    return PatchManifest.read(
        config.experiment_layout.patch_manifest_path(config.stage, config.split)
    ).records


def train_detector(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> list[EpochLoss]:
    train_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=DETECTOR_STAGE,
            split="train",
            subjects=train_subjects,
            patch_size=PatchSizes.DETECTOR,
            augmentation_factor=config.detector_augmentation_factor,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=DETECTOR_STAGE,
            split="validation",
            subjects=validation_subjects,
            patch_size=PatchSizes.DETECTOR,
            augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        )
    )
    detector = CandidateDetector().to(device)
    try:
        history = train_stage(
            detector,
            SegmentationTask(),
            SegmentationPatchDataset(train_records),
            SegmentationPatchDataset(validation_records),
            experiment_layout,
            DETECTOR_STAGE,
            config.detector_hyperparameters,
            config.num_workers,
            config.pin_memory,
        )
    finally:
        del detector
        utils.release_gpu_memory()
    return history


def train_teacher(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> list[EpochLoss]:
    teacher = CandidateDiscriminatorTeacher()
    initializer = CandidateDetector()
    try:
        core_io.load_model_weights(
            initializer,
            experiment_layout.best_checkpoint_path(DETECTOR_STAGE),
        )
        core_utils.initialize_teacher_from_detector(initializer, teacher)
    finally:
        del initializer

    teacher = teacher.to(device)

    train_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=TEACHER_STAGE,
            split="train",
            subjects=train_subjects,
            patch_size=PatchSizes.DISCRIMINATOR,
            augmentation_factor=config.discriminator_augmentation_factor,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=TEACHER_STAGE,
            split="validation",
            subjects=validation_subjects,
            patch_size=PatchSizes.DISCRIMINATOR,
            augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        )
    )
    try:
        history = train_stage(
            teacher,
            SegmentationClassificationTask(),
            SegmentationClassificationPatchDataset(train_records),
            SegmentationClassificationPatchDataset(validation_records),
            experiment_layout,
            TEACHER_STAGE,
            config.teacher_hyperparameters,
            config.num_workers,
            config.pin_memory,
        )
    finally:
        del teacher
        utils.release_gpu_memory()
    return history


def train_student(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> list[EpochLoss]:
    detector = CandidateDetector().to(device)
    try:
        core_io.load_model_weights(
            detector,
            experiment_layout.best_checkpoint_path(DETECTOR_STAGE),
        )
        train_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage=STUDENT_STAGE,
                split="train",
                subjects=train_subjects,
                patch_size=PatchSizes.DISCRIMINATOR,
                augmentation_factor=config.discriminator_augmentation_factor,
                probability_threshold=config.detector_candidate_threshold,
                detector=detector,
            )
        )
        validation_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage=STUDENT_STAGE,
                split="validation",
                subjects=validation_subjects,
                patch_size=PatchSizes.DISCRIMINATOR,
                augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
                probability_threshold=config.detector_candidate_threshold,
                detector=detector,
            )
        )
    finally:
        del detector
        utils.release_gpu_memory()

    student = CandidateDiscriminatorStudent().to(device)
    teacher = CandidateDiscriminatorTeacher().to(device)
    try:
        core_io.load_model_weights(
            teacher,
            experiment_layout.best_checkpoint_path(TEACHER_STAGE),
        )
        history = train_stage(
            student,
            KnowledgeDistillationClassificationTask(teacher),
            ClassificationPatchDataset(train_records),
            ClassificationPatchDataset(validation_records),
            experiment_layout,
            STUDENT_STAGE,
            config.student_hyperparameters,
            config.num_workers,
            config.pin_memory,
        )
    finally:
        del student
        del teacher
        utils.release_gpu_memory()
    return history
