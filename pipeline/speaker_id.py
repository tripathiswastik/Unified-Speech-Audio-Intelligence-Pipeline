"""Speaker Embedding & Identification Engine.
Produces noise-invariant speaker d-vectors for verification, identification, and diarization.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import BASE_DIR, SAMPLE_RATE
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.device import get_device


from pipeline.model import (
    TemporalStatsPooling,
    SpeakerEmbeddingNet,
    SESpeakerEmbeddingNet,
    ArcMarginProduct,
)


class SpeakerIdentifier:
    """Enrolls voiceprints and identifies speakers based on vectorized cosine similarity."""

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
        """Extract a 128-dimensional L2-normalized speaker embedding.
        
        For long utterances (>4.0s), computes windowed embeddings over 3-second segments
        and averages them, reducing memory footprint and improving speaker representation.
        """
        max_single_samples = int(4.0 * SAMPLE_RATE)
        if len(waveform) > max_single_samples:
            window_size = int(3.0 * SAMPLE_RATE)
            hop_size = int(1.5 * SAMPLE_RATE)
            window_embs = []
            with torch.no_grad():
                for st in range(0, len(waveform) - window_size + 1, hop_size):
                    chunk = waveform[st : st + window_size]
                    spec = compute_log_mel_spectrogram(chunk)
                    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    emb = self.model(spec_tensor)[0].cpu().numpy()
                    window_embs.append(emb)
            if window_embs:
                avg_emb = np.mean(window_embs, axis=0)
                norm = np.linalg.norm(avg_emb)
                return (avg_emb / max(norm, 1e-12)).astype(np.float32)

        spec = compute_log_mel_spectrogram(waveform)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.model(spec_tensor)[0].cpu().numpy()
        return emb.astype(np.float32)

    def extract_batch_embeddings(self, waveforms: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for multiple audio windows using streaming chunked batching."""
        if not waveforms:
            return np.empty((0, 128), dtype=np.float32)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(waveforms), batch_size):
                chunk_wavs = waveforms[i : i + batch_size]
                chunk_specs = [compute_log_mel_spectrogram(w) for w in chunk_wavs]
                batch_tensor = torch.from_numpy(np.stack(chunk_specs, axis=0)).unsqueeze(1).float().to(self.device)
                chunk_embs = self.model(batch_tensor).cpu().numpy()
                all_embeddings.append(chunk_embs)

        return np.concatenate(all_embeddings, axis=0)

    def extract_batch_embeddings_from_specs(
        self, specs: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings from pre-sliced spectrograms with streaming GPU batching."""
        if not specs:
            return np.empty((0, 128), dtype=np.float32)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(specs), batch_size):
                chunk = specs[i : i + batch_size]
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

    def identify(
        self,
        waveform: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """Identify which enrolled speaker matches using vectorized matrix search.

        Supports passing either raw audio waveform or precomputed 128-d voiceprint embedding.
        """
        if not self.enrolled_profiles or self._enrolled_matrix is None:
            return "Unknown_Speaker", 0.0

        if embedding is not None:
            query_emb = embedding
        elif waveform is not None:
            query_emb = self.extract_embedding(waveform)
        else:
            return "Unknown_Speaker", 0.0

        # Vectorized dot product against all enrolled profiles in one operation
        sims = np.dot(self._enrolled_matrix, query_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_speaker = self._enrolled_names[best_idx]

        if best_sim < threshold:
            return f"Unenrolled ({best_speaker} dist: {best_sim:.2f})", best_sim

        return best_speaker, best_sim

    def identify_batch(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Identify multiple speakers simultaneously from precomputed (N, 128) embeddings in a single GEMM."""
        if len(embeddings) == 0:
            return []
        if not self.enrolled_profiles or self._enrolled_matrix is None:
            return [("Unknown_Speaker", 0.0) for _ in range(len(embeddings))]

        # (N, num_enrolled) = (N, 128) @ (128, num_enrolled)
        sim_matrix = np.dot(embeddings, self._enrolled_matrix.T)
        results = []
        for i in range(len(embeddings)):
            sims = sim_matrix[i]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_speaker = self._enrolled_names[best_idx]
            if best_sim < threshold:
                results.append((f"Unenrolled ({best_speaker} dist: {best_sim:.2f})", best_sim))
            else:
                results.append((best_speaker, best_sim))
        return results

