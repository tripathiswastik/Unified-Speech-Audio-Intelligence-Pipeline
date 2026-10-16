"""Speaker Diarization Engine.
Partitions continuous multi-speaker audio into 'Who spoke when' temporal turns using Agglomerative Clustering.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from pipeline.config import SAMPLE_RATE
from pipeline.speaker_id import SpeakerEmbeddingNet, SpeakerIdentifier


class DiarizationSegment:
    def __init__(self, start: float, end: float, speaker: str, embedding: np.ndarray):
        self.start = round(start, 2)
        self.end = round(end, 2)
        self.speaker = speaker
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": f"{self.start:.2f}s",
            "end_time": f"{self.end:.2f}s",
            "speaker": self.speaker,
        }


class SpeakerDiarizer:
    """Sliding-window diarizer with energy-based VAD, batched neural embeddings, and fast clustering."""

    def __init__(
        self,
        embedding_net: Optional[SpeakerEmbeddingNet] = None,
        window_size_sec: float = 1.5,
        hop_size_sec: float = 0.5,
        vad_energy_threshold: float = 0.015,
        device: Optional[Any] = None,
    ):
        self.device = device
        self.identifier = SpeakerIdentifier(model=embedding_net, device=self.device)
        self.window_size = int(window_size_sec * SAMPLE_RATE)
        self.hop_size = int(hop_size_sec * SAMPLE_RATE)
        self.window_sec = window_size_sec
        self.hop_sec = hop_size_sec
        self.vad_thresh = vad_energy_threshold

    def diarize(
        self, waveform: np.ndarray, num_speakers: int = 2
    ) -> List[DiarizationSegment]:
        """Perform speaker diarization on a continuous waveform using batched inference."""
        total_samples = len(waveform)
        if total_samples < self.window_size:
            return []

        valid_windows = []
        window_meta = []

        # 1. Sliding Window & VAD Extraction
        for start_idx in range(0, total_samples - self.window_size + 1, self.hop_size):
            end_idx = start_idx + self.window_size
            window = waveform[start_idx:end_idx]

            # RMS Energy check (VAD)
            rms = np.sqrt(np.mean(window ** 2))
            if rms < self.vad_thresh:
                continue  # Skip silent / low-energy segment

            start_t = start_idx / SAMPLE_RATE
            end_t = end_idx / SAMPLE_RATE
            valid_windows.append(window)
            window_meta.append((start_t, end_t))

        if not valid_windows:
            return []

        # 2. Batched Neural Embedding Forward Pass (High-Throughput)
        X = self.identifier.extract_batch_embeddings(valid_windows, batch_size=32)
        n_clusters = min(num_speakers, len(X))

        # Optional dimensionality reduction for long audio (> 50 windows) to accelerate clustering
        X_clust = X
        if len(X) > 50 and X.shape[1] > 16:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=16, random_state=42)
            X_clust = svd.fit_transform(X)

        if n_clusters > 1:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="cosine",
                    linkage="average",
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="average",
                    **{"affinity": "cosine"},
                )
            labels = clustering.fit_predict(X_clust)
        else:
            labels = [0] * len(X)

        # 3. Assemble and Merge Consecutive Windows of the Same Speaker
        raw_segments = []
        for i, (st, et) in enumerate(window_meta):
            spk_label = f"Speaker_{labels[i]}"
            raw_segments.append(DiarizationSegment(st, et, spk_label, X[i]))

        merged_segments = []
        current = raw_segments[0]

        for next_seg in raw_segments[1:]:
            # If same speaker and overlapping/adjacent within hop size, extend
            if next_seg.speaker == current.speaker and next_seg.start <= current.end + 0.1:
                current.end = next_seg.end
            else:
                merged_segments.append(current)
                current = next_seg
        merged_segments.append(current)

        return merged_segments
