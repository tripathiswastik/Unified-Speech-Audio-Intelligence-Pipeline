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
        try:
            param = next(self.model.parameters())
            if param.device != self.device:
                self.model.to(self.device)
        except StopIteration:
            self.model.to(self.device)
        self.model.eval()
        self.enrolled_profiles: Dict[str, np.ndarray] = {}
        self._enrolled_matrix: Optional[np.ndarray] = None
        self._enrolled_names: List[str] = []

    def _update_enrolled_matrix(self):
        """Build a cached 2D matrix of enrolled voiceprints for fast vectorized search."""
        if not self.enrolled_profiles:
            self._enrolled_matrix = None
            self._enrolled_names = []
            return
        self._enrolled_names = list(self.enrolled_profiles.keys())
        self._enrolled_matrix = np.stack([self.enrolled_profiles[name] for name in self._enrolled_names], axis=0)

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Extract a 128-dimensional L2-normalized speaker embedding."""
        spec = compute_log_mel_spectrogram(waveform)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.model(spec_tensor)[0].cpu().numpy()
        return emb

    def extract_batch_embeddings(self, waveforms: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for multiple audio windows in batched forward passes."""
        if not waveforms:
            return np.empty((0, 128), dtype=np.float32)

        # Precompute spectrograms
        specs = [compute_log_mel_spectrogram(w) for w in waveforms]
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(specs), batch_size):
                chunk = specs[i : i + batch_size]
                # Stack into (B, 1, n_mels, time_steps)
                batch_tensor = torch.from_numpy(np.stack(chunk, axis=0)).unsqueeze(1).float().to(self.device)
                chunk_embs = self.model(batch_tensor).cpu().numpy()
                all_embeddings.append(chunk_embs)

        return np.concatenate(all_embeddings, axis=0)

    def enroll(self, speaker_name: str, audio_path: Path):
        """Enroll a speaker voiceprint into the identification registry."""
        waveform = load_and_preprocess_audio(audio_path)
        emb = self.extract_embedding(waveform)
        if speaker_name in self.enrolled_profiles:
            # Running average of voiceprint embeddings using direct NumPy normalization
            combined = self.enrolled_profiles[speaker_name] + emb
            norm = np.linalg.norm(combined)
            self.enrolled_profiles[speaker_name] = (combined / max(norm, 1e-12)).astype(np.float32)
        else:
            self.enrolled_profiles[speaker_name] = emb.astype(np.float32)
        self._update_enrolled_matrix()

    def identify(self, waveform: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
        """Identify which enrolled speaker matches the waveform using vectorized matrix search."""
        if not self.enrolled_profiles or self._enrolled_matrix is None:
            return "Unknown_Speaker", 0.0

        query_emb = self.extract_embedding(waveform)
        # Vectorized dot product against all enrolled profiles in one operation
        sims = np.dot(self._enrolled_matrix, query_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_speaker = self._enrolled_names[best_idx]

        if best_sim < threshold:
            return f"Unenrolled ({best_speaker}?)", best_sim

        return best_speaker, best_sim
