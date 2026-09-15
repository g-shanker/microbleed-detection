from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
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
    PatchRecord,
    PatchSizes,
    TrainingHyperparameters,
)
from ...core.engines.trainers import Trainer
from .. import utils
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    ManifestStatus,
    PatchManifest,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    TrainManifest,
    timestamp,
)
from . import patch

VALIDATION_AUGMENTATION_FACTOR = 1


def execute(config: TrainConfig) -> None:
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        dataset_layout.preprocessed_manifest_path()
    )
    train_subjects, validation_subjects, test_subjects = split_subjects(
        preprocessed_manifest.subjects,
        config.train_size,
        config.validation_size,
        config.test_size,
        config.seed,
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
        config.dataset_dir,
        config.device,
        config.seed,
        train_subjects,
        validation_subjects,
        test_subjects,
        config,
        detector_history,
        teacher_history,
        student_history,
    )


def split_subjects(
    subjects: list[PreprocessedSubject],
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int | None = None,
) -> tuple[
    list[PreprocessedSubject], list[PreprocessedSubject], list[PreprocessedSubject]
]:
    train_subjects, held_out_subjects = cast(
        tuple[list[PreprocessedSubject], list[PreprocessedSubject]],
        train_test_split(subjects, train_size=train_size, random_state=seed),
    )
    validation_subjects, test_subjects = cast(
        tuple[list[PreprocessedSubject], list[PreprocessedSubject]],
        train_test_split(
            held_out_subjects,
            train_size=validation_size / (validation_size + test_size),
            random_state=seed,
        ),
    )
    return train_subjects, validation_subjects, test_subjects


def write_train_manifest(
    experiment_layout: ExperimentLayout,
    dataset_dir: Path,
    device: str,
    seed: int | None,
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    test_subjects: list[PreprocessedSubject],
    config: TrainConfig,
    detector_history: list[EpochLoss],
    teacher_history: list[EpochLoss],
    student_history: list[EpochLoss],
) -> None:
    now = timestamp()
    TrainManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        dataset_dir=str(dataset_dir.resolve()),
        device=device,
        seed=seed,
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        train_subject_ids=[subject.subject_id for subject in train_subjects],
        validation_subject_ids=[
            subject.subject_id for subject in validation_subjects
        ],
        test_subject_ids=[subject.subject_id for subject in test_subjects],
        detector_candidate_threshold=config.detector_candidate_threshold,
        detector_augmentation_factor=config.detector_augmentation_factor,
        discriminator_augmentation_factor=config.discriminator_augmentation_factor,
        validation_augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        detector_patch_size=PatchSizes.DETECTOR,
        discriminator_patch_size=PatchSizes.DISCRIMINATOR,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        training_settings=config.training_settings,
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
    stage: str,
    hyperparameters: TrainingHyperparameters,
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
            stage="detector",
            split="train",
            subjects=train_subjects,
            patch_size=PatchSizes.DETECTOR,
            augmentation_factor=config.detector_augmentation_factor,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="detector",
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
            "detector",
            config.training_settings,
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
            experiment_layout.best_checkpoint_path("detector"),
        )
        core_utils.initialize_teacher_from_detector(initializer, teacher)
    finally:
        del initializer

    teacher = teacher.to(device)

    train_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="teacher",
            split="train",
            subjects=train_subjects,
            patch_size=PatchSizes.DISCRIMINATOR,
            augmentation_factor=config.discriminator_augmentation_factor,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="teacher",
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
            "teacher",
            config.training_settings,
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
            experiment_layout.best_checkpoint_path("detector"),
        )
        train_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage="student",
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
                stage="student",
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
            experiment_layout.best_checkpoint_path("teacher"),
        )
        history = train_stage(
            student,
            KnowledgeDistillationClassificationTask(teacher),
            ClassificationPatchDataset(train_records),
            ClassificationPatchDataset(validation_records),
            experiment_layout,
            "student",
            config.training_settings,
            config.num_workers,
            config.pin_memory,
        )
    finally:
        del student
        del teacher
        utils.release_gpu_memory()
    return history
