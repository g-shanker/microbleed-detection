import torch
import torch.nn as nn
import torch.nn.functional as F

POINTWISE_KERNEL_SIZE = 1
POOL_KERNEL_SIZE = 2
UPSAMPLE_KERNEL_SIZE = 2
UPSAMPLE_STRIDE = 2


class SingleConv(nn.Module):
    """
    convolution -> batch-normalization -> relu
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LinearBlock(nn.Module):
    """
    linear -> batch-normalization -> relu
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DoubleConv(nn.Module):
    """
    (convolution -> batch-normalization -> relu) twice
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int,
        kernel_size_2: int,
    ):
        super().__init__()

        intermediate_channels = out_channels
        padding_1 = (kernel_size_1 - 1) // 2
        padding_2 = (kernel_size_2 - 1) // 2

        self.layer = nn.Sequential(
            SingleConv(in_channels, intermediate_channels, kernel_size_1, padding_1),
            SingleConv(intermediate_channels, out_channels, kernel_size_2, padding_2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DownConv(nn.Module):
    """
    maxpool with kernel_size 2 -> (convolution -> batch_normalization -> relu) twice
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int,
        kernel_size_2: int,
        pool_padding: int,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool3d(POOL_KERNEL_SIZE, padding=pool_padding),
            DoubleConv(
                in_channels,
                out_channels,
                kernel_size_1,
                kernel_size_2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UpConv(nn.Module):
    """
    convolution transpose (upsampling) -> concatenation with skip connection ->
    (convolution -> batch-normalization -> relu) twice
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int,
        kernel_size_2: int,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            UPSAMPLE_KERNEL_SIZE,
            stride=UPSAMPLE_STRIDE,
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            kernel_size_1,
            kernel_size_2,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        x1 = self.upsample(x1)
        target_shape = x2.shape[2:]
        current_shape = x1.shape[2:]

        crop_starts = [
            max((current_size - target_size) // 2, 0)
            for current_size, target_size in zip(current_shape, target_shape)
        ]
        crop_ends = [
            start + min(current_size, target_size)
            for start, current_size, target_size in zip(
                crop_starts, current_shape, target_shape
            )
        ]
        x1 = x1[
            ...,
            crop_starts[0] : crop_ends[0],
            crop_starts[1] : crop_ends[1],
            crop_starts[2] : crop_ends[2],
        ]

        padding = [
            padding
            for target_size, current_size in reversed(
                list(zip(target_shape, x1.shape[2:]))
            )
            for padding in (
                max((target_size - current_size) // 2, 0),
                max(target_size - current_size, 0)
                - max((target_size - current_size) // 2, 0),
            )
        ]
        x1 = F.pad(x1, padding)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    convolution with kernel_size=1
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Conv3d(
            in_channels, out_channels, kernel_size=POINTWISE_KERNEL_SIZE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
