"""Modern 2D-CNN Architecture for Environmental Sound Classification on Mel-Spectrograms."""
import torch
import torch.nn as nn
from pipeline.config import CLASSES


class ConvBlock(nn.Module):
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


class AudioCNN(nn.Module):
    """Deep 2D-CNN for Audio Mel-Spectrogram Classification."""
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        # Input shape: (B, 1, 128, 173) for 4.0s @ 22050Hz, n_fft=2048, hop=512
        self.block1 = ConvBlock(1, 32, dropout_rate=0.1)
        self.block2 = ConvBlock(32, 64, dropout_rate=0.2)
        self.block3 = ConvBlock(64, 128, dropout_rate=0.2)
        self.block4 = ConvBlock(128, 256, dropout_rate=0.3)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
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
