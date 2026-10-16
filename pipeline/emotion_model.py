"""Speech Emotion Recognition (SER) Neural Network Architectures.
Includes:
  1. EmotionClassifierCNN (Baseline dual-pooling 2D-CNN)
  2. SEEmotionClassifierCNN (Fortified with Squeeze-and-Excitation channel attention)
  3. SSLEmotionClassifier (Pretrained Self-Supervised Learning wav2vec 2.0 / HuBERT adapter)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.emotion_dataset import EMOTIONS
from pipeline.model import SEBlock


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


class SEEmotionConvBlock(nn.Module):
    """Emotion convolutional block with Squeeze-and-Excitation channel recalibration."""
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.2, reduction: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ELU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ELU(inplace=True),
            SEBlock(out_c, reduction=reduction),
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


class SEEmotionClassifierCNN(nn.Module):
    """Upgraded SER architecture featuring Squeeze-and-Excitation frequency attention."""

    def __init__(self, num_classes: int = len(EMOTIONS)):
        super().__init__()
        self.block1 = SEEmotionConvBlock(1, 32, dropout=0.1, reduction=8)
        self.block2 = SEEmotionConvBlock(32, 64, dropout=0.2, reduction=8)
        self.block3 = SEEmotionConvBlock(64, 128, dropout=0.25, reduction=16)
        self.block4 = SEEmotionConvBlock(128, 256, dropout=0.3, reduction=16)

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


class SSLEmotionClassifier(nn.Module):
    """Pretrained Self-Supervised Learning (SSL) Adapter for Speech Emotion Recognition.
    
    Accepts raw speech waveforms or SSL transformer hidden states (wav2vec 2.0 / HuBERT),
    freezes the upstream acoustic representation, and trains an attentive pooling head.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = len(EMOTIONS),
        pretrained_model_name: Optional[str] = "facebook/wav2vec2-base",
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        self.encoder = None

        # Attempt to load Hugging Face transformer encoder if installed
        if pretrained_model_name:
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(pretrained_model_name)
                if self.freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
            except Exception:
                # Graceful fallback: model acts as downstream projection head on input embeddings
                self.encoder = None

        # Attentive temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is raw audio (B, T) and encoder is loaded:
        if self.encoder is not None and x.ndim == 2:
            with torch.set_grad_enabled(not self.freeze_encoder):
                outputs = self.encoder(x)
                h = outputs.last_hidden_state  # (B, T_frames, hidden_dim)
        elif x.ndim == 3:
            # Input is already extracted SSL feature representation (B, T, hidden_dim)
            h = x
        elif x.ndim == 4:
            # 2D Spectrogram input projected to SSL dimension
            B, C, F_dim, T_dim = x.shape
            h = x.squeeze(1).permute(0, 2, 1)  # (B, T, F_dim)
            if h.shape[-1] != self.hidden_dim:
                # Linear project to hidden_dim if needed
                if not hasattr(self, "_dim_proj") or self._dim_proj.in_features != h.shape[-1]:
                    self._dim_proj = nn.Linear(h.shape[-1], self.hidden_dim).to(x.device)
                h = self._dim_proj(h)

        # Attentive temporal pooling: sum_t(alpha_t * h_t)
        attn_weights = F.softmax(self.attention(h), dim=1)  # (B, T, 1)
        pooled = torch.sum(h * attn_weights, dim=1)         # (B, hidden_dim)

        return self.head(pooled)
