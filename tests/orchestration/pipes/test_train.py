from pathlib import Path

import torch
import torch.nn as nn

from microbleednet.core.common.models import CandidateDetector
from microbleednet.orchestration.configs import TrainConfig
from microbleednet.orchestration.layouts import DatasetLayout, ExperimentLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
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


def test_execute_runs_stages_with_fixed_training_values(
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
        ) -> Path:
            calls.append(
                (
                    name,
                    [subject.subject_id for subject in train_subjects],
                    [subject.subject_id for subject in validation_subjects],
                    _dependencies,
                )
            )
            return layout.best_checkpoint_path(checkpoint_name)

        return stage

    monkeypatch.setattr(
        train, "train_detector", record_stage("detector", "detector")
    )
    monkeypatch.setattr(
        train, "train_teacher", record_stage("teacher", "teacher")
    )
    monkeypatch.setattr(
        train, "train_student", record_stage("student", "student")
    )
    experiment_dir = tmp_path / "experiment"

    train.execute(
        TrainConfig(
            dataset_dir=dataset_dir,
            experiment_dir=experiment_dir,
        )
    )

    assert [call[0] for call in calls] == ["detector", "teacher", "student"]
    assert all(call[1:3] == calls[0][1:3] for call in calls)
    assert calls[2][3][-1] == train.DETECTOR_CANDIDATE_THRESHOLD


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
            self, model, task, best_checkpoint, settings
        ) -> None:
            trainer_calls.append(
                (model, task, best_checkpoint, settings)
            )

        def fit(self, train_loader, validation_loader) -> None:
            trainer_calls.append((train_loader, validation_loader))

    def extract(config):
        patch_calls.append(config)
        return [object(), object()]

    monkeypatch.setattr(train.patch, "execute", extract)
    monkeypatch.setattr(train, "SegmentationPatchDataset", FakeDataset)
    monkeypatch.setattr(
        train, "SegmentationClassificationPatchDataset", FakeDataset
    )
    monkeypatch.setattr(train, "ClassificationPatchDataset", FakeDataset)
    monkeypatch.setattr(train, "EqualBatchSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "SequentialSampler", lambda dataset: object())
    monkeypatch.setattr(train, "BatchSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "DataLoader", lambda *args, **kwargs: object())
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

    train.train_detector(subjects, subjects, layout, device)
    train.train_teacher(subjects, subjects, layout, device)
    train.train_student(
        subjects,
        subjects,
        layout,
        device,
        0.25,
    )

    assert [call.patch_size for call in patch_calls] == [48, 48, 24, 24, 24, 24]
    assert [call.augmentation_factor for call in patch_calls] == [10, 1, 5, 1, 5, 1]
    assert patch_calls[4].probability_threshold == 0.25
    assert patch_calls[4].detector is patch_calls[5].detector
    assert len(trainer_calls) == 6
    assert len(initialized) == 1
    assert [path for _, path in loaded] == [
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("teacher"),
    ]