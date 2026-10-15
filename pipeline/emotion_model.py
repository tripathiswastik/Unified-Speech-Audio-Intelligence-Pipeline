"""Speech Emotion Recognition (SER) Neural Network Architecture."""
import torch
import torch.nn as nn
from pipeline.emotion_dataset import EMOTIONS


class EmotionConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ELU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionClassifierCNN(nn.Module):
    """Deep Audio Classifier for 8-class Speech Emotion Recognition."""

    def __init__(self, num_classes: int = len(EMOTIONS)):
        super().__init__()
        self.block1 = EmotionConvBlock(1, 32, dropout=0.1)
        self.block2 = EmotionConvBlock(32, 64, dropout=0.2)
        self.block3 = EmotionConvBlock(64, 128, dropout=0.25)
        self.block4 = EmotionConvBlock(128, 256, dropout=0.3)

        # Concatenate AvgPool and MaxPool to capture sustained intonation and explosive bursts
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        avg_f = self.avg_pool(x)
        max_f = self.max_pool(x)
        feat = torch.cat([avg_f, max_f], dim=1)
        return self.head(feat)
