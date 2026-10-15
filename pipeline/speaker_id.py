"""Speaker Embedding & Identification Engine.
Produces noise-invariant speaker d-vectors for verification, identification, and diarization.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import BASE_DIR
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram


class TemporalStatsPooling(nn.Module):
    """Computes mean and standard deviation across the temporal dimension (x-vector standard)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T) -> pool over F and T
        B, C, F, T = x.shape
        x_flat = x.view(B, C * F, T)
        mean = torch.mean(x_flat, dim=-1)
        std = torch.std(x_flat, dim=-1)
        return torch.cat([mean, std], dim=1)


class SpeakerEmbeddingNet(nn.Module):
    """Deep Speaker Verification Network producing normalized 128-d voiceprint embeddings."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
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
        # Stats pooling over reduced frequency and time bins
        # 128 channels * 16 freq bins = 2048 dims * 2 (mean+std) = 4096 dims
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


from pipeline.device import get_device


class SpeakerIdentifier:
    """Enrolls voiceprints and identifies speakers based on cosine similarity."""

    def __init__(self, model: Optional[SpeakerEmbeddingNet] = None, device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device(verbose=False)
        self.model = model if model is not None else SpeakerEmbeddingNet()
        self.model.to(self.device)
        self.model.eval()
        self.enrolled_profiles: Dict[str, np.ndarray] = {}

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Extract a 128-dimensional L2-normalized speaker embedding."""
        spec = compute_log_mel_spectrogram(waveform)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.model(spec_tensor)[0].cpu().numpy()
        return emb

    def enroll(self, speaker_name: str, audio_path: Path):
        """Enroll a speaker voiceprint into the identification registry."""
        waveform = load_and_preprocess_audio(audio_path)
        emb = self.extract_embedding(waveform)
        if speaker_name in self.enrolled_profiles:
            # Running average of voiceprint embeddings
            self.enrolled_profiles[speaker_name] = F.normalize(
                torch.from_numpy(self.enrolled_profiles[speaker_name] + emb), p=2, dim=0
            ).numpy()
        else:
            self.enrolled_profiles[speaker_name] = emb

    def identify(self, waveform: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
        """Identify which enrolled speaker matches the waveform."""
        if not self.enrolled_profiles:
            return "Unknown_Speaker", 0.0

        query_emb = self.extract_embedding(waveform)
        best_speaker = "Unknown_Speaker"
        best_sim = -1.0

        for name, profile_emb in self.enrolled_profiles.items():
            sim = float(np.dot(query_emb, profile_emb))
            if sim > best_sim:
                best_sim = sim
                best_speaker = name

        if best_sim < threshold:
            return f"Unenrolled ({best_speaker}?)", best_sim

        return best_speaker, best_sim
