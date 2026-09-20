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
    TrainStageManifest,
    content_fingerprint,
)
from . import patch

VALIDATION_AUGMENTATION_FACTOR = 1


def execute(config: TrainConfig) -> None:
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


def write_train_manifest(
    experiment_layout: ExperimentLayout,
    config: TrainConfig,
    split_manifest_fingerprint: str,
    status: ManifestStatus,
) -> None:
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
        detector_patch_size=PatchSizes.DETECTOR,
        discriminator_patch_size=PatchSizes.DISCRIMINATOR,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        detector_hyperparameters=config.detector_hyperparameters,
        teacher_hyperparameters=config.teacher_hyperparameters,
        student_hyperparameters=config.student_hyperparameters,
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
        history=[],
    ).write(experiment_layout.stage_manifest_path(stage))
    if resume and trainer.latest_checkpoint.is_file():
        trainer.load_latest_checkpoint()
    history = trainer.fit(train_loader, validation_loader, f"Training {stage}")
    TrainStageManifest(
        status=ManifestStatus.COMPLETE,
        stage=stage,
        history=history,
    ).write(experiment_layout.stage_manifest_path(stage))
    trainer.latest_checkpoint.unlink(missing_ok=True)


def completed_stage(
    experiment_layout: ExperimentLayout, stage: StageName, resume: bool
) -> bool:
    if not resume:
        return False
    stage_manifest_path = experiment_layout.stage_manifest_path(stage)
    if not stage_manifest_path.is_file():
        return False
    stage_manifest = TrainStageManifest.read(stage_manifest_path)
    return stage_manifest.status is ManifestStatus.COMPLETE


def extract_patch_records(config: BasePatchConfig) -> list[PatchRecord]:
    patch.execute(config)
    manifest_path = config.experiment_layout.patch_manifest_path(
        config.stage, config.split
    )
    manifest = PatchManifest.read(manifest_path)
    if manifest.status is not ManifestStatus.COMPLETE:
        raise ValueError(
            f"manifest at {manifest_path} has status {manifest.status.value!r}; "
            "a consumer may only read a complete manifest."
        )
    return manifest.records


def train_detector(
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    experiment_layout: ExperimentLayout,
    device: torch.device,
    config: TrainConfig,
) -> None:
    if completed_stage(
        experiment_layout, DETECTOR_STAGE, config.resume
    ):
        return
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
    if completed_stage(
        experiment_layout, TEACHER_STAGE, config.resume
    ):
        return
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
    if completed_stage(
        experiment_layout, STUDENT_STAGE, config.resume
    ):
        return
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
