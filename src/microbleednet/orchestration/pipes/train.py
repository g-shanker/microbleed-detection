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
from ...core.datamodels import EpochLoss, PatchRecord, TrainingSettings
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
    PatchManifest,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    TrainManifest,
    timestamp,
)
from . import patch

TRAIN_SIZE = 0.7
DETECTOR_CANDIDATE_THRESHOLD = 0.5
DETECTOR_AUGMENTATION_FACTOR = 10
DISCRIMINATOR_AUGMENTATION_FACTOR = 5
VALIDATION_AUGMENTATION_FACTOR = 1
DETECTOR_PATCH_SIZE = 48
DISCRIMINATOR_PATCH_SIZE = 24
NUM_WORKERS = 0
PIN_MEMORY = False


def execute(config: TrainConfig) -> None:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        dataset_layout.preprocessed_manifest_path()
    )
    train_subjects, validation_subjects = split_subjects(preprocessed_manifest.subjects)
    settings = TrainingSettings()
    device = torch.device(config.device)
    detector_history = train_detector(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        settings,
    )
    teacher_history = train_teacher(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        settings,
    )
    student_history = train_student(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        settings,
    )
    write_train_manifest(
        experiment_layout,
        config.dataset_dir,
        config.device,
        train_subjects,
        validation_subjects,
        settings,
        detector_history,
        teacher_history,
        student_history,
    )


def split_subjects(
    subjects: list[PreprocessedSubject],
) -> tuple[list[PreprocessedSubject], list[PreprocessedSubject]]:
    train_subjects, validation_subjects = cast(
        tuple[list[PreprocessedSubject], list[PreprocessedSubject]],
        train_test_split(subjects, train_size=TRAIN_SIZE),
    )
    return train_subjects, validation_subjects


def write_train_manifest(
    experiment_layout: ExperimentLayout,
    dataset_dir: Path,
    device: str,
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    settings: TrainingSettings,
    detector_history: list[EpochLoss],
    teacher_history: list[EpochLoss],
    student_history: list[EpochLoss],
) -> None:
    now = timestamp()
    TrainManifest(
        status="complete",
        created_at=now,
        updated_at=now,
        dataset_dir=str(dataset_dir.resolve()),
        device=device,
        train_size=TRAIN_SIZE,
        train_subject_ids=[subject.subject_id for subject in train_subjects],
        validation_subject_ids=[
            subject.subject_id for subject in validation_subjects
        ],
        detector_candidate_threshold=DETECTOR_CANDIDATE_THRESHOLD,
        detector_augmentation_factor=DETECTOR_AUGMENTATION_FACTOR,
        discriminator_augmentation_factor=DISCRIMINATOR_AUGMENTATION_FACTOR,
        validation_augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        detector_patch_size=DETECTOR_PATCH_SIZE,
        discriminator_patch_size=DISCRIMINATOR_PATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        training_settings=settings,
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
    settings: TrainingSettings,
) -> list[EpochLoss]:
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=EqualBatchSampler(
            train_dataset.patches,
            batch_size=settings.batch_size,
        ),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(validation_dataset),
            batch_size=settings.batch_size,
            drop_last=False,
        ),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return Trainer(
        model,
        task,
        best_checkpoint=experiment_layout.best_checkpoint_path(stage),
        settings=settings,
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
    settings: TrainingSettings,
) -> list[EpochLoss]:
    train_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="detector",
            split="train",
            subjects=train_subjects,
            patch_size=DETECTOR_PATCH_SIZE,
            augmentation_factor=DETECTOR_AUGMENTATION_FACTOR,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="detector",
            split="validation",
            subjects=validation_subjects,
            patch_size=DETECTOR_PATCH_SIZE,
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
            settings,
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
    settings: TrainingSettings,
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
            patch_size=DISCRIMINATOR_PATCH_SIZE,
            augmentation_factor=DISCRIMINATOR_AUGMENTATION_FACTOR,
        )
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage="teacher",
            split="validation",
            subjects=validation_subjects,
            patch_size=DISCRIMINATOR_PATCH_SIZE,
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
            settings,
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
    settings: TrainingSettings,
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
                patch_size=DISCRIMINATOR_PATCH_SIZE,
                augmentation_factor=DISCRIMINATOR_AUGMENTATION_FACTOR,
                probability_threshold=DETECTOR_CANDIDATE_THRESHOLD,
                detector=detector,
            )
        )
        validation_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage="student",
                split="validation",
                subjects=validation_subjects,
                patch_size=DISCRIMINATOR_PATCH_SIZE,
                augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
                probability_threshold=DETECTOR_CANDIDATE_THRESHOLD,
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
            settings,
        )
    finally:
        del student
        del teacher
        utils.release_gpu_memory()
    return history
