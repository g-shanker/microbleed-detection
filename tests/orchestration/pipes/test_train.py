from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from microbleednet.constants import (
    DETECTOR_PATCH_SIZE,
    DETECTOR_STAGE,
    STUDENT_STAGE,
    TEACHER_STAGE,
)
from microbleednet.core.common.models import CandidateDetector
from microbleednet.core.datamodels import PatchRecord
from microbleednet.orchestration.configs import NonOverlappingPatchConfig, TrainConfig
from microbleednet.orchestration.layouts import (
    DatasetLayout,
    ExperimentLayout,
)
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PatchManifest,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    PreprocessedVariant,
    SplitManifest,
    content_fingerprint,
    timestamp,
)
from microbleednet.orchestration.pipes import train


def _hyperparameters(batch_size: int, max_epochs: int = 100) -> train.Hyperparameters:
    return train.Hyperparameters(
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=20,
        learning_rate=1e-3,
        adam_epsilon=1e-4,
        learning_rate_factor=0.1,
        learning_rate_period=2,
        minimum_learning_rate=1e-6,
        weight_decay=0.0,
        minimum_improvement=0.0,
        use_amp=False,
    )


def _subjects(
    count: int, directory: Path, augmentation_factor: int = 1
) -> list[PreprocessedSubject]:
    directory.mkdir(parents=True, exist_ok=True)
    subjects = []
    for index in range(count):
        variants = []
        for variant_index in range(augmentation_factor):
            volume_path = directory / f"volume-{index}-{variant_index}.nii.gz"
            mask_path = directory / f"mask-{index}-{variant_index}.nii.gz"
            volume_path.touch()
            mask_path.touch()
            variants.append(
                PreprocessedVariant(
                    volume_path=str(volume_path),
                    mask_path=str(mask_path),
                    frst_path=str(volume_path),
                )
            )
        subjects.append(
            PreprocessedSubject(
                subject_id=f"subject-{index}",
                original_volume_path=str(variants[0].volume_path),
                bounding_box=((0, 1), (0, 1), (0, 1)),
                variants=variants,
            )
        )
    return subjects


def test_execute_runs_stages_with_configured_training_values(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    subjects = _subjects(10, tmp_path / "inputs", augmentation_factor=10)
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=subjects,
        augmentation_factor=10,
        raw_manifest_fingerprint="test-raw-manifest",
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

    detector_hyperparameters = _hyperparameters(4, 7)
    teacher_hyperparameters = _hyperparameters(4, 8)
    student_hyperparameters = _hyperparameters(6, 9)
    config = TrainConfig(
        dataset_dir=dataset_dir,
        experiment_dir=experiment_dir,
        device="cpu",
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
            self, model, task, best_checkpoint, hyperparameters, **kwargs
        ) -> None:
            trainer_calls.append(
                (model, task, best_checkpoint, hyperparameters)
            )
            self.latest_checkpoint = kwargs["latest_checkpoint"]

        def fit(self, train_loader, validation_loader, description) -> list:
            trainer_calls.append((train_loader, validation_loader, description))
            return []

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
        train.utils,
        "load_stage_checkpoint",
        lambda model, path, stage: loaded.append((model, path, stage)),
    )
    monkeypatch.setattr(
        train.core_utils,
        "initialize_teacher_from_detector",
        lambda detector, teacher: initialized.append((detector, teacher)),
    )
    monkeypatch.setattr(train.utils, "release_gpu_memory", lambda: None)
    layout = ExperimentLayout(experiment_dir=tmp_path)
    subjects = _subjects(2, tmp_path / "inputs", augmentation_factor=8)
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
        augmentation_factor=8,
        raw_manifest_fingerprint="test-raw-manifest",
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
        device="cpu",
        detector_candidate_threshold=0.7,
        detector_augmentation_factor=8,
        discriminator_augmentation_factor=4,
        num_workers=2,
        pin_memory=True,
        detector_hyperparameters=_hyperparameters(4),
        teacher_hyperparameters=_hyperparameters(4),
        student_hyperparameters=_hyperparameters(6),
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
        "detector",
        "teacher",
        "student",
    ]
    assert all(call["num_workers"] == 2 for call in loader_calls)
    assert all(call["pin_memory"] is True for call in loader_calls)
    assert len(initialized) == 1
    assert [path for _, path, _ in loaded] == [
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("detector"),
        layout.best_checkpoint_path("teacher"),
    ]
    assert [stage for _, _, stage in loaded] == [
        "detector",
        "detector",
        "teacher",
    ]


def test_train_stage_resumes_from_latest_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    resumed = []
    fit_calls = []

    class FakeDataset:
        def __init__(self, patches) -> None:
            self.patches = patches

        def __len__(self) -> int:
            return len(self.patches)

    class FakeTrainer:
        def __init__(
            self, model, task, best_checkpoint, hyperparameters, latest_checkpoint
        ) -> None:
            latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            latest_checkpoint.write_bytes(b"checkpoint")
            self.latest_checkpoint = latest_checkpoint

        def load_latest_checkpoint(self) -> None:
            resumed.append(True)

        def fit(self, train_loader, validation_loader, description) -> list:
            fit_calls.append(description)
            return []

    monkeypatch.setattr(train, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "EqualBatchSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "SequentialSampler", lambda dataset: object())
    monkeypatch.setattr(train, "BatchSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "Trainer", FakeTrainer)

    experiment_layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    train.train_stage(
        nn.Linear(1, 1),
        SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        FakeDataset([object()]),  # pyright: ignore[reportArgumentType]
        FakeDataset([object()]),  # pyright: ignore[reportArgumentType]
        experiment_layout,
        DETECTOR_STAGE,
            _hyperparameters(2, 1),
        num_workers=0,
        pin_memory=False,
        resume=True,
    )

    assert resumed == [True]
    assert fit_calls == ["detector"]


def test_completed_stage_respects_resume_and_manifest_status(
    tmp_path: Path, monkeypatch
) -> None:
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    stage_manifest_path = layout.stage_manifest_path(DETECTOR_STAGE)
    stage_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    assert train.is_stage_completed(layout, DETECTOR_STAGE, resume=False) is False
    assert train.is_stage_completed(layout, DETECTOR_STAGE, resume=True) is False

    stage_manifest_path.touch()
    monkeypatch.setattr(
        train.TrainStageManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.RUNNING),
    )
    assert train.is_stage_completed(layout, DETECTOR_STAGE, resume=True) is False

    monkeypatch.setattr(
        train.TrainStageManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.COMPLETE),
    )
    assert train.is_stage_completed(layout, DETECTOR_STAGE, resume=True) is True


def test_extract_patch_records_rejects_incomplete_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    config = NonOverlappingPatchConfig(
        experiment_layout=ExperimentLayout(experiment_dir=tmp_path / "experiment"),
        stage=DETECTOR_STAGE,
        split="train",
        subjects=_subjects(1, tmp_path / "inputs"),
        patch_size=DETECTOR_PATCH_SIZE,
        augmentation_factor=1,
    )
    monkeypatch.setattr(train.patch, "execute", lambda _: None)
    monkeypatch.setattr(
        train.PatchManifest,
        "read",
        lambda _: SimpleNamespace(status=ManifestStatus.RUNNING, records=[]),
    )

    with pytest.raises(ValueError) as error:
        train.extract_patch_records(config)

    message = str(error.value)
    assert "incomplete patch output" in message
    assert "Rerun the train command with resume set to false" in message


def test_extract_patch_records_reuses_compatible_manifest_on_resume(
    tmp_path: Path, monkeypatch
) -> None:
    subject = _subjects(1, tmp_path / "inputs")[0]
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    config = NonOverlappingPatchConfig(
        experiment_layout=layout,
        stage=DETECTOR_STAGE,
        split="train",
        subjects=[subject],
        patch_size=DETECTOR_PATCH_SIZE,
        augmentation_factor=1,
    )
    paths = [tmp_path / name for name in ("volume.npy", "mask.npy", "frst.npy")]
    for path in paths:
        path.touch()
    record = PatchRecord(
        volume_path=str(paths[0]),
        mask_path=str(paths[1]),
        frst_path=str(paths[2]),
        patch_index=0,
        has_microbleed=True,
    )
    PatchManifest(
        status=ManifestStatus.COMPLETE,
        stage=config.stage,
        split=config.split,
        subject_ids=[subject.subject_id],
        patch_size=config.patch_size,
        augmentation_factor=config.augmentation_factor,
        records=[record],
    ).write(layout.patch_manifest_path(config.stage, config.split))
    monkeypatch.setattr(
        train.patch,
        "execute",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected extraction")),
    )

    assert train.extract_patch_records(config, resume=True) == [record]


def test_extract_patch_records_regenerates_missing_cached_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    subject = _subjects(1, tmp_path / "inputs")[0]
    layout = ExperimentLayout(experiment_dir=tmp_path / "experiment")
    config = NonOverlappingPatchConfig(
        experiment_layout=layout,
        stage=DETECTOR_STAGE,
        split="train",
        subjects=[subject],
        patch_size=DETECTOR_PATCH_SIZE,
        augmentation_factor=1,
    )
    manifest_path = layout.patch_manifest_path(config.stage, config.split)
    stale_record = PatchRecord(
        volume_path=str(tmp_path / "missing-volume.npy"),
        mask_path=str(tmp_path / "missing-mask.npy"),
        frst_path=str(tmp_path / "missing-frst.npy"),
        patch_index=0,
        has_microbleed=True,
    )
    PatchManifest(
        status=ManifestStatus.COMPLETE,
        stage=config.stage,
        split=config.split,
        subject_ids=[subject.subject_id],
        patch_size=config.patch_size,
        augmentation_factor=config.augmentation_factor,
        records=[stale_record],
    ).write(manifest_path)
    regenerated_paths = [
        tmp_path / name
        for name in (
            "regenerated-volume.npy",
            "regenerated-mask.npy",
            "regenerated-frst.npy",
        )
    ]
    for path in regenerated_paths:
        path.touch()
    regenerated_record = stale_record.model_copy(
        update={
            "volume_path": str(regenerated_paths[0]),
            "mask_path": str(regenerated_paths[1]),
            "frst_path": str(regenerated_paths[2]),
        }
    )
    extraction_calls = []

    def regenerate(patch_config: NonOverlappingPatchConfig) -> None:
        extraction_calls.append(patch_config)
        PatchManifest(
            status=ManifestStatus.COMPLETE,
            stage=patch_config.stage,
            split=patch_config.split,
            subject_ids=[subject.subject_id],
            patch_size=patch_config.patch_size,
            augmentation_factor=patch_config.augmentation_factor,
            records=[regenerated_record],
        ).write(manifest_path)

    monkeypatch.setattr(train.patch, "execute", regenerate)

    assert train.extract_patch_records(config, resume=True) == [regenerated_record]
    assert extraction_calls == [config]


@pytest.mark.parametrize(
    ("stage_function", "loader_attr"),
    [
        ("train_detector", None),
            ("train_teacher", "load_stage_checkpoint"),
            ("train_student", "load_stage_checkpoint"),
    ],
)
def test_stage_functions_return_early_when_stage_already_complete(
    tmp_path: Path,
    monkeypatch,
    stage_function: str,
    loader_attr: str | None,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    now = timestamp()
    PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=[],
        raw_manifest_fingerprint="test-raw-manifest",
    ).write(DatasetLayout(dataset_dir=dataset_dir).preprocessed_manifest_path())
    monkeypatch.setattr(train, "is_stage_completed", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        train,
        "extract_patch_records",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected extraction")
            ),
    )
    monkeypatch.setattr(
        train,
        "train_stage",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected training")
            ),
    )
    if loader_attr is not None:
        monkeypatch.setattr(
                train.utils,
            loader_attr,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected checkpoint load")
            ),
        )

    stage = getattr(train, stage_function)
    stage(
        _subjects(1, tmp_path / "inputs"),
        _subjects(1, tmp_path / "validation"),
        ExperimentLayout(experiment_dir=tmp_path / "experiment"),
        torch.device("cpu"),
        SimpleNamespace(resume=True),
    )