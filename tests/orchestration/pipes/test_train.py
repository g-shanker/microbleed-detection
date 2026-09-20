from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.configs import TrainConfig
from microbleednet.orchestration.layouts import (
    DETECTOR_STAGE,
    STUDENT_STAGE,
    TEACHER_STAGE,
    DatasetLayout,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    SplitManifest,
    content_fingerprint,
    timestamp,
)
from microbleednet.orchestration.pipes import train


def _subjects(count: int, directory: Path) -> list[PreprocessedSubject]:
    directory.mkdir(parents=True, exist_ok=True)
    subjects = []
    for index in range(count):
        volume_path = directory / f"volume-{index}.nii.gz"
        mask_path = directory / f"mask-{index}.nii.gz"
        volume_path.touch()
        mask_path.touch()
        subjects.append(
            PreprocessedSubject(
                subject_id=f"subject-{index}",
                variants=[
                    PreprocessedVariant(
                        volume_path=str(volume_path),
                        mask_path=str(mask_path),
                        frst_path=str(volume_path),
                    )
                ],
            )
        )
    return subjects


def test_execute_runs_stages_with_configured_training_values(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    subjects = _subjects(10, tmp_path / "inputs")
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=subjects,
        augmentation_factor=10,
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    calls: list[tuple[str, list[str], list[str], tuple[object, ...]]] = []

    def record_stage(name: str, checkpoint_name: str):
        def stage(
            train_subjects: list[PreprocessedSubject],
            validation_subjects: list[PreprocessedSubject],
            layout: ExperimentLayout,
            _device: torch.device,
            *_dependencies,
        ) -> list:
            calls.append(
                (
                    name,
                    [subject.subject_id for subject in train_subjects],
                    [subject.subject_id for subject in validation_subjects],
                    _dependencies,
                )
            )
            return []

        return stage

    monkeypatch.setattr(
        train, "train_detector", record_stage(DETECTOR_STAGE, DETECTOR_STAGE)
    )
    monkeypatch.setattr(
        train, "train_teacher", record_stage(TEACHER_STAGE, TEACHER_STAGE)
    )
    monkeypatch.setattr(
        train, "train_student", record_stage(STUDENT_STAGE, STUDENT_STAGE)
    )
    experiment_dir = tmp_path / "experiment"
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    )
    split_manifest = SplitManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        dataset_dir=str(dataset_dir.resolve()),
        preprocessed_manifest_fingerprint=content_fingerprint(preprocessed_manifest),
        seed=7,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        train_subject_ids=[subject.subject_id for subject in subjects[:6]],
        validation_subject_ids=[subject.subject_id for subject in subjects[6:8]],
        test_subject_ids=[subject.subject_id for subject in subjects[8:]],
    )
    split_manifest.write(
        ExperimentLayout(experiment_dir=experiment_dir).split_manifest_path()
    )

    detector_hyperparameters = train.Hyperparameters(batch_size=3, max_epochs=7)
    teacher_hyperparameters = train.Hyperparameters(batch_size=4, max_epochs=8)
    student_hyperparameters = train.Hyperparameters(batch_size=5, max_epochs=9)
    config = TrainConfig(
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        seed=42,
        detector_candidate_threshold=0.7,
        detector_augmentation_factor=8,
        discriminator_augmentation_factor=4,
        num_workers=2,
        pin_memory=True,
        detector_hyperparameters=detector_hyperparameters,
        teacher_hyperparameters=teacher_hyperparameters,
        student_hyperparameters=student_hyperparameters,
    )
    train.execute(config)

    train_manifest = train.TrainManifest.read(
        ExperimentLayout(experiment_dir=experiment_dir).train_manifest_path()
    )
    assert [call[0] for call in calls] == [
        DETECTOR_STAGE,
        TEACHER_STAGE,
        STUDENT_STAGE,
    ]
    assert all(call[1:3] == calls[0][1:3] for call in calls)
    assert calls[2][3] == (config,)
    assert train_manifest.dataset_dir == str(dataset_dir.resolve())
    assert train_manifest.device == "cpu"
    assert train_manifest.seed == config.seed
    assert (
        train_manifest.detector_candidate_threshold
        == config.detector_candidate_threshold
    )
    assert train_manifest.detector_augmentation_factor == 8
    assert train_manifest.discriminator_augmentation_factor == 4
    assert train_manifest.num_workers == 2
    assert train_manifest.pin_memory is True
    assert train_manifest.detector_hyperparameters == detector_hyperparameters
    assert train_manifest.teacher_hyperparameters == teacher_hyperparameters
    assert train_manifest.student_hyperparameters == student_hyperparameters
    assert (
        ExperimentLayout(experiment_dir=experiment_dir).train_manifest_path()
        == experiment_dir / "manifests" / "train.json"
    )

    unseeded_config = config.model_copy(
        update={
            "seed": None,
            "experiment_dir": tmp_path / "unseeded-experiment",
        }
    )
    split_manifest.write(
        ExperimentLayout(
            experiment_dir=unseeded_config.experiment_dir
        ).split_manifest_path()
    )
    train.execute(unseeded_config)
    unseeded_manifest = train.TrainManifest.read(
        ExperimentLayout(
            experiment_dir=unseeded_config.experiment_dir
        ).train_manifest_path()
    )
    assert unseeded_manifest.seed is None


def test_stage_functions_apply_fixed_training_recipe(
    tmp_path: Path, monkeypatch
) -> None:
    patch_calls = []
    trainer_calls = []
    loaded = []
    initialized = []

    class FakeDataset:
        def __init__(self, records) -> None:
            self.patches = records

    class FakeTrainer:
        def __init__(
            self, model, task, best_checkpoint, hyperparameters
        ) -> None:
            trainer_calls.append(
                (model, task, best_checkpoint, hyperparameters)
            )

        def fit(self, train_loader, validation_loader, description) -> None:
            trainer_calls.append((train_loader, validation_loader, description))

    def extract(config):
        patch_calls.append(config)
        manifest_path = config.experiment_layout.patch_manifest_path(
            config.stage, config.split
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.touch()
        return None

    monkeypatch.setattr(
        train.PatchManifest,
            "read",
            lambda path: SimpleNamespace(
                status=ManifestStatus.COMPLETE,
                records=[object(), object()],
            ),
    )

    monkeypatch.setattr(train.patch, "execute", extract)
    monkeypatch.setattr(train, "SegmentationPatchDataset", FakeDataset)
    monkeypatch.setattr(
        train, "SegmentationClassificationPatchDataset", FakeDataset
    )
    monkeypatch.setattr(train, "ClassificationPatchDataset", FakeDataset)
    monkeypatch.setattr(train, "EqualBatchSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "SequentialSampler", lambda dataset: object())
    monkeypatch.setattr(train, "BatchSampler", lambda *args, **kwargs: object())
    loader_calls = []

    def data_loader(*args, **kwargs):
        loader_calls.append(kwargs)
        return object()

    monkeypatch.setattr(train, "DataLoader", data_loader)
    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "CandidateDetector", CandidateDetector)
    monkeypatch.setattr(
        train, "CandidateDiscriminatorTeacher", lambda: nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        train, "CandidateDiscriminatorStudent", lambda: nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        train.core_io,
        "load_model_weights",
        lambda model, path: loaded.append((model, path)),
    )
    monkeypatch.setattr(
        train.core_utils,
        "initialize_teacher_from_detector",
        lambda detector, teacher: initialized.append((detector, teacher)),
    )
    monkeypatch.setattr(train.utils, "release_gpu_memory", lambda: None)
    layout = ExperimentLayout(experiment_dir=tmp_path)
    subjects = _subjects(2, tmp_path / "inputs")
    device = torch.device("cpu")
    detector_checkpoint = layout.best_checkpoint_path("detector")
    detector_checkpoint.parent.mkdir(parents=True)
    detector_checkpoint.touch()

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=[],
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    preprocessed_manifest = PreprocessedDatasetManifest.read(
        DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path()
    )
    SplitManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        dataset_dir=str(dataset_dir.resolve()),
        preprocessed_manifest_fingerprint=content_fingerprint(preprocessed_manifest),
        seed=None,
        train_size=0.7,
        validation_size=0.1,
        test_size=0.2,
        train_subject_ids=[],
        validation_subject_ids=[],
        test_subject_ids=[],
    ).write(
        ExperimentLayout(experiment_dir=tmp_path / "experiment").split_manifest_path()
    )
    config = TrainConfig(
        dataset_dir=dataset_dir,
        experiment_dir=tmp_path / "experiment",
        detector_candidate_threshold=0.7,
        detector_augmentation_factor=8,
        discriminator_augmentation_factor=4,
        num_workers=2,
        pin_memory=True,
        detector_hyperparameters=train.Hyperparameters(batch_size=3),
        teacher_hyperparameters=train.Hyperparameters(batch_size=4),
        student_hyperparameters=train.Hyperparameters(batch_size=5),
    )
    train.train_detector(subjects, subjects, layout, device, config)
    train.train_teacher(subjects, subjects, layout, device, config)
    train.train_student(
        subjects,
        subjects,
        layout,
        device,
        config,
    )

    assert [call.patch_size for call in patch_calls] == [48, 48, 24, 24, 24, 24]
    assert [call.augmentation_factor for call in patch_calls] == [8, 1, 4, 1, 4, 1]
    assert patch_calls[4].probability_threshold == config.detector_candidate_threshold
    assert patch_calls[4].detector is patch_calls[5].detector
    assert len(trainer_calls) == 6
    assert [call[2] for call in trainer_calls[1::2]] == [
        "Training detector",
        "Training teacher",
        "Training student",
    ]
    assert all(call["num_workers"] == 2 for call in loader_calls)
    assert all(call["pin_memory"] is True for call in loader_calls)
    assert len(initialized) == 1
    assert [path for _, path in loaded] == [
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("teacher"),
    ]