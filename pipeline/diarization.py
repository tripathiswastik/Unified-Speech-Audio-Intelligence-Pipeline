"""Speaker Diarization Engine.
Partitions continuous multi-speaker audio into 'Who spoke when' temporal turns using Agglomerative Clustering.
"""
from typing import List, Dict, Any, Tuple
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
    """Sliding-window diarizer with energy-based VAD and Agglomerative Clustering."""

    def __init__(
        self,
        embedding_net: SpeakerEmbeddingNet = None,
        window_size_sec: float = 1.5,
        hop_size_sec: float = 0.5,
        vad_energy_threshold: float = 0.015,
    ):
        self.identifier = SpeakerIdentifier(model=embedding_net)
        self.window_size = int(window_size_sec * SAMPLE_RATE)
        self.hop_size = int(hop_size_sec * SAMPLE_RATE)
        self.window_sec = window_size_sec
        self.hop_sec = hop_size_sec
        self.vad_thresh = vad_energy_threshold

    def diarize(
        self, waveform: np.ndarray, num_speakers: int = 2
    ) -> List[DiarizationSegment]:
        """Perform speaker diarization on a continuous waveform."""
        total_samples = len(waveform)
        if total_samples < self.window_size:
            return []

        window_embeddings = []
        window_meta = []

        # 1. Sliding Window & VAD Extraction
        for start_idx in range(0, total_samples - self.window_size + 1, self.hop_size):
            end_idx = start_idx + self.window_size
            window = waveform[start_idx:end_idx]

            # RMS Energy check (VAD)
            rms = np.sqrt(np.mean(window ** 2))
            if rms < self.vad_thresh:
                continue  # Skip silent / low-energy segment

            emb = self.identifier.extract_embedding(window)
            window_embeddings.append(emb)
            start_t = start_idx / SAMPLE_RATE
            end_t = end_idx / SAMPLE_RATE
            window_meta.append((start_t, end_t, emb))

        if not window_embeddings:
            return []

        # 2. Agglomerative Clustering on Cosine Distance
        X = np.array(window_embeddings)
        n_clusters = min(num_speakers, len(X))

        if n_clusters > 1:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(X)
        else:
            labels = [0] * len(X)

        # 3. Assemble and Merge Consecutive Windows of the Same Speaker
        raw_segments = []
        for i, (st, et, emb) in enumerate(window_meta):
            spk_label = f"Speaker_{labels[i]}"
            raw_segments.append(DiarizationSegment(st, et, spk_label, emb))

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
