import logging

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from ...constants import (
    DETECTOR_PATCH_SIZE,
    DETECTOR_STAGE,
    DISCRIMINATOR_PATCH_SIZE,
    SKIPPING_COMPLETED_STAGE_MESSAGE,
    STUDENT_STAGE,
    TEACHER_STAGE,
    VALIDATION_AUGMENTATION_FACTOR,
)
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
    Hyperparameters,
    PatchRecord,
)
from ...core.engines.trainers import Trainer
from ...errors import ApplicationError
from .. import utils
from ..configs import (
    BasePatchConfig,
    NonOverlappingPatchConfig,
    TargetCenteredPatchConfig,
    TrainConfig,
)
from ..layouts import (
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
    TrainStageManifest,
    content_fingerprint,
)
from . import patch

logger = logging.getLogger(__name__)

def execute(config: TrainConfig) -> None:
    """Run the detector, teacher, and student training stages in sequence."""
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    experiment_layout = ExperimentLayout(experiment_dir=config.experiment_dir)
    dataset_layout = DatasetLayout(dataset_dir=config.dataset_dir)
    preprocessed_manifest_path = dataset_layout.preprocessed_manifest_path()
    preprocessed_manifest = PreprocessedDatasetManifest.read(preprocessed_manifest_path)
    split_manifest_path = experiment_layout.split_manifest_path()
    split_manifest = SplitManifest.read(split_manifest_path)
    train_subjects = utils.resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.train_subject_ids
    )
    validation_subjects = utils.resolve_subjects(
        preprocessed_manifest.subjects, split_manifest.validation_subject_ids
    )
    device = torch.device(config.device)
    logger.info(
        "Starting training pipeline in %s on %s (resume=%s): "
        "detector, teacher, student",
        config.experiment_dir,
        device,
        config.resume,
    )
    write_train_manifest(
        experiment_layout,
        config,
        content_fingerprint(split_manifest),
        ManifestStatus.RUNNING,
    )
    train_detector(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    train_teacher(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    train_student(
        train_subjects,
        validation_subjects,
        experiment_layout,
        device,
        config,
    )
    write_train_manifest(
        experiment_layout,
        config,
        content_fingerprint(split_manifest),
        ManifestStatus.COMPLETE,
    )
    logger.info(
        "Training pipeline complete; manifest written to %s",
        experiment_layout.train_manifest_path(),
    )


def write_train_manifest(
    experiment_layout: ExperimentLayout,
    config: TrainConfig,
    split_manifest_fingerprint: str,
    status: ManifestStatus,
) -> None:
    """Persist training provenance and lifecycle status for an experiment."""
    TrainManifest(
        status=status,
        dataset_dir=utils.resolve_path_string(config.dataset_dir),
        split_manifest_fingerprint=split_manifest_fingerprint,
        device=config.device,
        seed=config.seed,
        detector_candidate_threshold=config.detector_candidate_threshold,
        detector_augmentation_factor=config.detector_augmentation_factor,
        discriminator_augmentation_factor=config.discriminator_augmentation_factor,
        validation_augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        detector_patch_size=DETECTOR_PATCH_SIZE,
        discriminator_patch_size=DISCRIMINATOR_PATCH_SIZE,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
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
    resume: bool = False,
) -> None:
    """Fit one model stage and persist its lifecycle and training history."""
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
    trainer = Trainer(
        model,
        task,
        best_checkpoint=experiment_layout.best_checkpoint_path(stage),
        hyperparameters=hyperparameters,
        latest_checkpoint=experiment_layout.latest_checkpoint_path(stage),
    )
    TrainStageManifest(
        status=ManifestStatus.RUNNING,
        stage=stage,
        hyperparameters=hyperparameters,
        history=[],
    ).write(experiment_layout.stage_manifest_path(stage))
    logger.info(
        "Starting %s: train_patches=%d validation_patches=%d "
        "batch_size=%d learning_rate=%g max_epochs=%d patience=%d",
        stage,
        len(train_dataset.patches),
        len(validation_dataset.patches),
        hyperparameters.batch_size,
        hyperparameters.learning_rate,
        hyperparameters.max_epochs,
        hyperparameters.patience,
    )
    if resume and trainer.latest_checkpoint.is_file():
        logger.info("Resuming %s from %s", stage, trainer.latest_checkpoint)
        trainer.load_latest_checkpoint()
    history = trainer.fit(train_loader, validation_loader, str(stage))
    TrainStageManifest(
        status=ManifestStatus.COMPLETE,
        stage=stage,
        hyperparameters=hyperparameters,
        history=history,
    ).write(experiment_layout.stage_manifest_path(stage))
    trainer.latest_checkpoint.unlink(missing_ok=True)
    logger.info("Completed %s stage", stage)
def is_stage_completed(
    experiment_layout: ExperimentLayout, stage: StageName, resume: bool
) -> bool:
    """Check whether resume mode can skip a previously completed stage."""
    if not resume:
        return False
    stage_manifest_path = experiment_layout.stage_manifest_path(stage)
    if not stage_manifest_path.is_file():
        return False
    stage_manifest = TrainStageManifest.read(stage_manifest_path)
    return stage_manifest.status is ManifestStatus.COMPLETE


def extract_patch_records(
    config: BasePatchConfig, resume: bool = False
) -> list[PatchRecord]:
    """Reuse compatible patches or extract and validate a fresh patch set."""
    manifest_path = config.experiment_layout.patch_manifest_path(
        config.stage, config.split
    )

    if resume and manifest_path.is_file():
        existing_manifest = PatchManifest.read(manifest_path)
        if patch.manifest_matches_config(existing_manifest, config):
            logger.info(
                "Reusing %d %s %s patches from %s",
                len(existing_manifest.records),
                config.stage,
                config.split,
                manifest_path,
            )
            return existing_manifest.records

    patch.execute(config)
    manifest = PatchManifest.read(manifest_path)
    if manifest.status is not ManifestStatus.COMPLETE:
        raise ApplicationError(
            category="Manifest",
            summary="Cannot train from incomplete patch output",
            cause=f"Patch manifest status is '{manifest.status.value}'",
            fix=(
                "Rerun the train command with resume set to false to regenerate "
                "patch artifacts"
            ),
            context={"path": str(manifest_path)},
        )
    return manifest.records


def train_detector(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Train the candidate detector from non-overlapping subject patches."""
    if is_stage_completed(
        experiment_layout, DETECTOR_STAGE, config.resume
    ):
        logger.info(SKIPPING_COMPLETED_STAGE_MESSAGE, DETECTOR_STAGE)
        return
    train_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=DETECTOR_STAGE,
            split="train",
            subjects=train_subjects,
            patch_size=DETECTOR_PATCH_SIZE,
            augmentation_factor=config.detector_augmentation_factor,
        ),
        resume=config.resume,
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=DETECTOR_STAGE,
            split="validation",
            subjects=validation_subjects,
            patch_size=DETECTOR_PATCH_SIZE,
            augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        ),
        resume=config.resume,
    )
    detector = CandidateDetector().to(device)
    try:
        train_stage(
            detector,
            SegmentationTask(),
            SegmentationPatchDataset(train_records),
            SegmentationPatchDataset(validation_records),
            experiment_layout,
            DETECTOR_STAGE,
            config.detector_hyperparameters,
            config.num_workers,
            config.pin_memory,
            config.resume,
        )
    finally:
        del detector
        utils.release_gpu_memory()


def train_teacher(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Initialize and train the teacher discriminator from detector features."""
    if is_stage_completed(
        experiment_layout, TEACHER_STAGE, config.resume
    ):
        logger.info(SKIPPING_COMPLETED_STAGE_MESSAGE, TEACHER_STAGE)
        return
    teacher = CandidateDiscriminatorTeacher()
    initializer = CandidateDetector()
    try:
        utils.load_stage_checkpoint(
            initializer,
            experiment_layout.best_checkpoint_path(DETECTOR_STAGE),
            DETECTOR_STAGE,
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
            patch_size=DISCRIMINATOR_PATCH_SIZE,
            augmentation_factor=config.discriminator_augmentation_factor,
        ),
        resume=config.resume,
    )
    validation_records = extract_patch_records(
        NonOverlappingPatchConfig(
            experiment_layout=experiment_layout,
            stage=TEACHER_STAGE,
            split="validation",
            subjects=validation_subjects,
            patch_size=DISCRIMINATOR_PATCH_SIZE,
            augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
        ),
        resume=config.resume,
    )
    try:
        train_stage(
            teacher,
            SegmentationClassificationTask(),
            SegmentationClassificationPatchDataset(train_records),
            SegmentationClassificationPatchDataset(validation_records),
            experiment_layout,
            TEACHER_STAGE,
            config.teacher_hyperparameters,
            config.num_workers,
            config.pin_memory,
            config.resume,
        )
    finally:
        del teacher
        utils.release_gpu_memory()


def train_student(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Train the student discriminator on detector-centered candidate patches."""
    if is_stage_completed(
        experiment_layout, STUDENT_STAGE, config.resume
    ):
        logger.info(SKIPPING_COMPLETED_STAGE_MESSAGE, STUDENT_STAGE)
        return
    detector = CandidateDetector().to(device)
    try:
        utils.load_stage_checkpoint(
            detector,
            experiment_layout.best_checkpoint_path(DETECTOR_STAGE),
            DETECTOR_STAGE,
        )
        train_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage=STUDENT_STAGE,
                split="train",
                subjects=train_subjects,
                patch_size=DISCRIMINATOR_PATCH_SIZE,
                augmentation_factor=config.discriminator_augmentation_factor,
                probability_threshold=config.detector_candidate_threshold,
                detector=detector,
            ),
            resume=config.resume,
        )
        validation_records = extract_patch_records(
            TargetCenteredPatchConfig(
                experiment_layout=experiment_layout,
                stage=STUDENT_STAGE,
                split="validation",
                subjects=validation_subjects,
                patch_size=DISCRIMINATOR_PATCH_SIZE,
                augmentation_factor=VALIDATION_AUGMENTATION_FACTOR,
                probability_threshold=config.detector_candidate_threshold,
                detector=detector,
            ),
            resume=config.resume,
        )
    finally:
        del detector
        utils.release_gpu_memory()

    student = CandidateDiscriminatorStudent().to(device)
    teacher = CandidateDiscriminatorTeacher().to(device)
    try:
        utils.load_stage_checkpoint(
            teacher,
            experiment_layout.best_checkpoint_path(TEACHER_STAGE),
            TEACHER_STAGE,
        )
        train_stage(
            student,
            KnowledgeDistillationClassificationTask(teacher),
            ClassificationPatchDataset(train_records),
            ClassificationPatchDataset(validation_records),
            experiment_layout,
            STUDENT_STAGE,
            config.student_hyperparameters,
            config.num_workers,
            config.pin_memory,
            config.resume,
        )
    finally:
        del student
        del teacher
        utils.release_gpu_memory()
