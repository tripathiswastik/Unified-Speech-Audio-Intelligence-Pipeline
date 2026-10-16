"""Modern 2D-CNN & SE-ResNet Architectures for Environmental Sound Classification.
Includes Squeeze-and-Excitation (SE) channel attention, Depthwise-Separable Convolutions,
and Additive Angular Margin (ArcFace) loss modules.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.config import CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.
    
    Adaptively recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between convolutional channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-Separable 2D Convolution to shrink parameter count while maintaining capacity."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """Baseline dual-convolution residual block."""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SEConvBlock(nn.Module):
    """Convolutional block fortified with Squeeze-and-Excitation channel attention."""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2, reduction: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(out_channels, reduction=reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AudioCNN(nn.Module):
    """Deep 2D-CNN for Audio Mel-Spectrogram Classification."""
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.block1 = ConvBlock(1, 32, dropout_rate=0.1)
        self.block2 = ConvBlock(32, 64, dropout_rate=0.2)
        self.block3 = ConvBlock(64, 128, dropout_rate=0.2)
        self.block4 = ConvBlock(128, 256, dropout_rate=0.3)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class SEAudioCNN(nn.Module):
    """Upgraded AudioCNN featuring Squeeze-and-Excitation Channel Attention.
    
    Dynamically weights Mel frequency bands and temporal dynamics to prioritize
    informative acoustic events while suppressing stationary background noise.
    """
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.block1 = SEConvBlock(1, 32, dropout_rate=0.1, reduction=8)
        self.block2 = SEConvBlock(32, 64, dropout_rate=0.2, reduction=8)
        self.block3 = SEConvBlock(64, 128, dropout_rate=0.2, reduction=16)
        self.block4 = SEConvBlock(128, 256, dropout_rate=0.3, reduction=16)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class ArcMarginProduct(nn.Module):
    """Additive Angular Margin (ArcFace) Loss Head.
    
    Forces the network to maximize inter-class angular margin on the hypersphere,
    tightening speaker cluster representations and boosting verification accuracy.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.30,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = float(np.cos(margin))
        self.sin_m = float(np.sin(margin))
        self.th = float(np.cos(np.pi - margin))
        self.mm = float(np.sin(np.pi - margin) * margin)

    def forward(self, input_features: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # L2-normalize input embeddings and classifier weights
        cosine = F.linear(F.normalize(input_features, p=2, dim=-1), F.normalize(self.weight, p=2, dim=-1))
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for target class angular margin injection
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output


class TemporalStatsPooling(nn.Module):
    """Computes mean and standard deviation along the temporal dimension for variable-length speech."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T) -> collapse C & F, then compute mean and std across T
        B, C, F, T = x.shape
        x_flat = x.view(B, C * F, T)
        mean = torch.mean(x_flat, dim=-1)
        std = torch.std(x_flat, dim=-1)
        return torch.cat([mean, std], dim=1)


class SpeakerEmbeddingNet(nn.Module):
    """Deep Speaker Verification Network producing normalized 128-d voiceprint embeddings.

    Supports 'baseline', 'se_resnet', and 'depthwise' backbones.
    Default 'baseline' preserves exact 1-to-1 state_dict layer compatibility with existing checkpoints.
    """

    def __init__(self, embedding_dim: int = 128, backbone: str = "baseline"):
        super().__init__()
        self.backbone_type = backbone
        self.embedding_dim = embedding_dim

        if backbone == "depthwise":
            self.conv = nn.Sequential(
                DepthwiseSeparableConv2d(1, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),

                DepthwiseSeparableConv2d(32, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),

                DepthwiseSeparableConv2d(64, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )
        elif backbone == "se_resnet":
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                SEBlock(32, reduction=8),
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SEBlock(64, reduction=8),
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SEBlock(128, reduction=16),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )
        else:
            # Baseline architecture: exactly identical layers for checkpoint loading
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            )

        self.stats_pool = TemporalStatsPooling()
        self.projection = nn.Sequential(
            nn.Linear(128 * 16 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        stats = self.stats_pool(feat)
        embedding = self.projection(stats)
        return F.normalize(embedding, p=2, dim=-1)


class SESpeakerEmbeddingNet(SpeakerEmbeddingNet):
    """Convenience alias for SE-ResNet fortified SpeakerEmbeddingNet."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__(embedding_dim=embedding_dim, backbone="se_resnet")

