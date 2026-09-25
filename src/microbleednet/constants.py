import re
from pathlib import Path

# Model layers and architecture
POINTWISE_KERNEL_SIZE = 1
POOL_KERNEL_SIZE = 2
UPSAMPLE_KERNEL_SIZE = 2
UPSAMPLE_STRIDE = 2
INPUT_CHANNELS = 2
OUTPUT_CLASSES = 2
LEVEL_CHANNELS = (3, 64, 128, 256)
CLASSIFIER_FEATURES = 1024
CLASSIFIER_HIDDEN_NODES = 128
CLASSIFIER_OUTPUT_NODES = 32
DROPOUT_RATE = 0.2
WEIGHT_INIT_STD = 0.05
BIAS_INIT_VALUE = 0.1
DETECTOR_PATCH_SIZE = 48
DISCRIMINATOR_PATCH_SIZE = 24

# Losses
DICE_SMOOTH = 1.0
FOREGROUND_CLASS = 1
DETECTOR_CLASS_WEIGHTS = (1.0, 10.0)
DISTILLATION_ALPHA = 0.4
DISTILLATION_BETA = 0.6
DISTILLATION_TEMPERATURE = 4.0

# Augmentation and preprocessing
TRANSLATION_OFFSET_RANGE = (-15, 15)
NOISE_VARIANCE_RANGE = (0.01, 0.04)
BLUR_SIGMA_RANGE = (0.1, 0.2)
AVAILABLE_TRANSFORMATIONS = ("translate", "noise", "blur")
FRST_RADII = (2, 3, 4, 6)
FRST_ALPHA = 2
FRST_FACTOR_STD = 0.1
FRANGI_SIGMAS = (0.5, 1.2, 0.2)
FRANGI_ALPHA = 0.9
FRANGI_BETA = 20
FRANGI_BLACK_RIDGES = False
CLUSTERER_N_CLUSTERS = 2
CLUSTERER_RANDOM_STATE = 0
MINIMUM_VESSEL_ECCENTRICITY = 0.9
MAXIMUM_VESSEL_SOLIDITY = 0.5
VESSEL_CONNECTIVITY = 1
INVERTED_MODALITIES = frozenset({"T2*-GRE", "SWI"})

# Core processing and training
AMP_DTYPE_NAME = "float16"
COMPONENT_CONNECTIVITY = 3
VALIDATION_AUGMENTATION_FACTOR = 1

# Inference
VARIANT_INDEX = 0
DETECTOR_THRESHOLD = 0.5
STUDENT_THRESHOLD = 0.5
MINIMUM_VOLUME_MM3 = 2.5
MAXIMUM_ELLIPTICITY = 0.2
MINIMUM_BRAIN_DISTANCE_MM = 5.0
PREPROCESSING_AUGMENTATION_FACTOR = 1

# Dataset and artifact naming
SUBJECT_ID_PLACEHOLDER = "{subject_id}"
SOURCE_ID_PLACEHOLDER = "{source_id}"
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_NIFTI_SUFFIX = ".nii.gz"
DETECTOR_STAGE = "detector"
TEACHER_STAGE = "teacher"
STUDENT_STAGE = "student"
RAW_MANIFEST_PATH = Path("manifests/raw.json")
PREPROCESSED_MANIFEST_PATH = Path("manifests/preprocessed.json")
PREPROCESSED_VOLUMES_PATH = Path("preprocessed/volumes")
PREPROCESSED_MASKS_PATH = Path("preprocessed/masks")
PREPROCESSED_FRST_PATH = Path("preprocessed/frst")
TRAIN_MANIFEST_PATH = Path("manifests/train.json")
SPLIT_MANIFEST_PATH = Path("manifests/split.json")
PATCH_MANIFEST_TEMPLATE = Path("manifests/patch_{stage}_{split}.json")
BEST_CHECKPOINT_TEMPLATE = Path("train/{stage}/checkpoints/best_model.pth")
LATEST_CHECKPOINT_TEMPLATE = Path("train/{stage}/checkpoints/latest_model.pth")
STAGE_MANIFEST_TEMPLATE = Path("train/{stage}/manifest.json")
PATCH_DIR_TEMPLATE = Path("train/{stage}/patches/{split}")
INFERENCE_MANIFEST_PATH = Path("manifests/infer.json")
EVALUATION_MANIFEST_PATH = Path("manifests/evaluate.json")
INFERENCE_OUTPUT_TEMPLATE = Path("infer/{subject_id}/detections.nii.gz")
RAW_MANIFEST_LABEL = "raw manifest"
PREPROCESSED_MANIFEST_LABEL = "preprocessed manifest"
SPLIT_MANIFEST_LABEL = "split manifest"
TRAIN_MANIFEST_LABEL = "train manifest"

# CLI presentation
CLI_HELP = (
	"Index, preprocess, train, infer, and evaluate cerebral microbleed "
	"detection models."
)
INDEX_DATA_COMMAND_HELP = (
	"Index one raw NIfTI source and optional masks into a dataset manifest. "
	"Run once per source before preprocessing."
)
PREPROCESS_COMMAND_HELP = (
	"Preprocess every indexed subject and write image, mask, and FRST variants "
	"for splitting and training."
)
SPLIT_COMMAND_HELP = (
	"Assign preprocessed subjects to train, validation, and test sets and write "
	"the experiment split manifest."
)
TRAIN_COMMAND_HELP = (
	"Train detector, teacher, and student stages from a prepared split, writing "
	"patches, checkpoints, and manifests."
)
INFER_COMMAND_HELP = (
	"Apply explicit detector and student checkpoints to an indexed dataset and "
	"write source-space binary detection masks."
)
EVALUATE_COMMAND_HELP = (
	"Run inference and lesion-level scoring for an experiment test split or an "
	"explicitly configured dataset."
)
DESCRIBE_COMMAND_HELP = (
	"Show TOML keys, defaults, and descriptions for a pipeline command."
)
DEVICE_FIELD_DESCRIPTION = "Torch device string, e.g. 'cpu' or 'cuda'."

SKIPPING_COMPLETED_STAGE_MESSAGE = "Skipping completed %s stage"