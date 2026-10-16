"""Speaker Diarization Engine.
Partitions continuous multi-speaker audio into 'Who spoke when' temporal turns
using Voice Activity Detection (VAD), batched neural embeddings, and Agglomerative Clustering.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
    """Sliding-window diarizer with VAD, batched neural embeddings, and fast clustering."""

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
        self, waveform: np.ndarray, num_speakers: Optional[int] = 2
    ) -> List[DiarizationSegment]:
        """Perform speaker diarization on a continuous waveform using batched inference.
        
        Args:
            waveform: 1D numpy array representing continuous audio.
            num_speakers: Expected number of speakers. If None or <= 0, automatically
                          estimates speaker count via silhouette scoring.
        """
        total_samples = len(waveform)
        if total_samples < self.window_size:
            return []

        raw_windows = []
        raw_meta = []
        is_speech = []

        # 1. Sliding Window & Initial Energy VAD
        for start_idx in range(0, total_samples - self.window_size + 1, self.hop_size):
            end_idx = start_idx + self.window_size
            window = waveform[start_idx:end_idx]
            rms = float(np.sqrt(np.mean(window ** 2)))
            speech_active = rms >= self.vad_thresh

            raw_windows.append(window)
            raw_meta.append((start_idx / SAMPLE_RATE, end_idx / SAMPLE_RATE))
            is_speech.append(speech_active)

        if not raw_windows:
            return []

        # 2. VAD Hangover Smoothing (Bridge brief pauses < 0.3s)
        smoothed_speech = list(is_speech)
        for i in range(1, len(is_speech) - 1):
            if not is_speech[i] and is_speech[i - 1] and is_speech[i + 1]:
                smoothed_speech[i] = True

        valid_windows = []
        window_meta = []
        for i, active in enumerate(smoothed_speech):
            if active:
                valid_windows.append(raw_windows[i])
                window_meta.append(raw_meta[i])

        if not valid_windows:
            return []

        # 3. Batched Neural Embedding Extraction
        X = self.identifier.extract_batch_embeddings(valid_windows, batch_size=32)

        # 4. Optional dimensionality reduction for long sessions (> 50 windows)
        X_clust = X
        if len(X) > 50 and X.shape[1] > 16:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=16, random_state=42)
            X_clust = svd.fit_transform(X)

        # 5. Determine Number of Speaker Clusters
        if num_speakers is None or num_speakers <= 0:
            if len(X) < 4:
                n_clusters = 1
            else:
                best_k = 2
                best_score = -1.0
                max_k = min(6, len(X))
                for k in range(2, max_k):
                    try:
                        clu = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                    except TypeError:
                        clu = AgglomerativeClustering(n_clusters=k, linkage="average", **{"affinity": "cosine"})
                    labs = clu.fit_predict(X_clust)
                    if len(set(labs)) > 1:
                        score = float(silhouette_score(X_clust, labs, metric="cosine"))
                        if score > best_score:
                            best_score = score
                            best_k = k
                n_clusters = best_k if best_score > 0.15 else 1
        else:
            n_clusters = min(num_speakers, len(X))

        # 6. Cluster Window Embeddings
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

        # 7. Assemble and Merge Adjacent Windows of the Same Speaker
        raw_segments = []
        for i, (st, et) in enumerate(window_meta):
            spk_label = f"Speaker_{labels[i]}"
            raw_segments.append(DiarizationSegment(st, et, spk_label, X[i]))

        merged_segments = []
        current = raw_segments[0]

        for next_seg in raw_segments[1:]:
            # If same speaker and adjacent within hop size + hangover tolerance (0.35s), extend
            if next_seg.speaker == current.speaker and next_seg.start <= current.end + 0.35:
                current.end = next_seg.end
            else:
                merged_segments.append(current)
                current = next_seg
        merged_segments.append(current)

        return merged_segments
