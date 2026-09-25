import torch
import torch.nn as nn

from ...constants import (
    BIAS_INIT_VALUE,
    CLASSIFIER_FEATURES,
    CLASSIFIER_HIDDEN_NODES,
    CLASSIFIER_OUTPUT_NODES,
    DROPOUT_RATE,
    INPUT_CHANNELS,
    LEVEL_CHANNELS,
    OUTPUT_CLASSES,
    WEIGHT_INIT_STD,
)
from . import layers


def weight_init(model):
    """Applies truncated normal initialization."""
    if isinstance(model, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        # PyTorch has a built-in truncated normal initializer
        nn.init.trunc_normal_(model.weight, std=WEIGHT_INIT_STD)
        if model.bias is not None:
            nn.init.constant_(model.bias, BIAS_INIT_VALUE)


class CandidateDetector(nn.Module):
    def __init__(self):
        """Build the candidate segmentation network."""
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.segmentor = Segmentor()

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict segmentation logits for candidate volumes."""
        features = self.feature_extractor(x)
        logits = self.segmentor(features)
        return logits


class CandidateDiscriminatorTeacher(nn.Module):
    def __init__(self):
        """Build the teacher network with segmentation and classification heads."""
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.segmentor = Segmentor()
        self.classifier = Classifier()

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict segmentation and classification logits for a volume."""
        features = self.feature_extractor(x)
        segmentation_logits = self.segmentor(features)
        classification_logits = self.classifier(features)
        return segmentation_logits, classification_logits


class CandidateDiscriminatorStudent(nn.Module):
    def __init__(self):
        """Build the compact student classification network."""
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict classification logits for a candidate volume."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class FeatureExtractor(nn.Module):
    def __init__(self):
        """Build the shared multiscale 3D feature encoder."""
        super().__init__()

        self.in_conv = layers.SingleConv(INPUT_CHANNELS, LEVEL_CHANNELS[0], 1, 0)
        self.conv_1 = layers.DoubleConv(
            LEVEL_CHANNELS[0],
            LEVEL_CHANNELS[1],
            3,
            3,
        )
        self.down_1 = layers.DownConv(
            LEVEL_CHANNELS[1], LEVEL_CHANNELS[2], 3, 3, pool_padding=0
        )
        self.down_2 = layers.DownConv(
            LEVEL_CHANNELS[2], LEVEL_CHANNELS[3], 3, 3, pool_padding=0
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode an input volume into multiscale feature maps."""
        x0 = self.in_conv(x)
        x1 = self.conv_1(x0)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)

        return {"x1": x1, "x2": x2, "x3": x3}


class Segmentor(nn.Module):
    def __init__(self):
        """Build the decoder that maps multiscale features to segmentation logits."""
        super().__init__()

        self.up_2 = layers.UpConv(LEVEL_CHANNELS[3], LEVEL_CHANNELS[2], 3, 3)
        self.up_1 = layers.UpConv(LEVEL_CHANNELS[2], LEVEL_CHANNELS[1], 3, 3)
        self.out_conv = layers.OutConv(LEVEL_CHANNELS[1], OUTPUT_CLASSES)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode multiscale features into segmentation logits."""
        x1 = features["x1"]
        x2 = features["x2"]
        x3 = features["x3"]

        x = self.up_2(x3, x2)
        x = self.up_1(x, x1)
        logits = self.out_conv(x)

        return logits


class Classifier(nn.Module):
    def __init__(self):
        """Build the convolutional and dense candidate classifier."""
        super().__init__()

        self.in_conv = layers.SingleConv(LEVEL_CHANNELS[3], LEVEL_CHANNELS[2], 1, 0)
        self.down_1 = layers.DownConv(
            LEVEL_CHANNELS[2], LEVEL_CHANNELS[2], 3, 3, pool_padding=0
        )
        self.down_2 = layers.DownConv(
            LEVEL_CHANNELS[2],
            LEVEL_CHANNELS[2],
            3,
            3,
            pool_padding=1,
        )
        self.fc_1 = layers.LinearBlock(CLASSIFIER_FEATURES, CLASSIFIER_HIDDEN_NODES)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc_2 = layers.LinearBlock(
            CLASSIFIER_HIDDEN_NODES,
            CLASSIFIER_OUTPUT_NODES,
        )
        self.fc_3 = nn.Linear(CLASSIFIER_OUTPUT_NODES, OUTPUT_CLASSES)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Classify a candidate from its deepest encoded features."""
        x3 = features["x3"]
        x = self.in_conv(x3)
        x = self.down_1(x)
        x = self.down_2(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        logits = self.fc_3(x)
        return logits
