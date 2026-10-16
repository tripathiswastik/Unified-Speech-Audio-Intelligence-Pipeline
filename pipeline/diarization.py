"""Speaker Diarization Engine.
Partitions continuous multi-speaker audio into 'Who spoke when' temporal turns
using Voice Activity Detection (VAD), full-file Mel spectrogram slicing,
batched neural embeddings, and Agglomerative Clustering.
Enforces strictly non-overlapping chronological speaker timelines.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from pipeline.config import SAMPLE_RATE, HOP_LENGTH
from pipeline.audio_loader import compute_log_mel_spectrogram
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
    """High-throughput diarizer with full-file spectrogram slicing and strict timeline invariants."""

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
        self.window_frames = max(1, int(round(self.window_size / HOP_LENGTH)))

    def diarize(
        self, waveform: np.ndarray, num_speakers: Optional[int] = 2
    ) -> List[DiarizationSegment]:
        """Perform speaker diarization with full-file STFT reuse and timeline boundary resolution."""
        total_samples = len(waveform)
        if total_samples < int(0.25 * SAMPLE_RATE):
            return []

        # Handle short audio below single window size
        if total_samples < self.window_size:
            pad_width = self.window_size - total_samples
            padded_wave = np.pad(waveform, (0, pad_width), mode="constant")
            rms = float(np.sqrt(np.mean(padded_wave ** 2)))
            if rms < self.vad_thresh:
                return []
            emb = self.identifier.extract_embedding(padded_wave)
            return [DiarizationSegment(0.0, total_samples / SAMPLE_RATE, "Speaker_0", emb)]

        # Step 1: Compute full-file Log-Mel Spectrogram once to eliminate redundant STFTs
        full_spec = compute_log_mel_spectrogram(waveform)
        total_spec_frames = full_spec.shape[1]

        raw_meta = []
        is_speech = []
        raw_spec_slices = []

        last_start = 0
        # Step 2: Sliding Window Energy VAD & Spectrogram Frame Slicing
        for start_idx in range(0, total_samples - self.window_size + 1, self.hop_size):
            last_start = start_idx
            end_idx = start_idx + self.window_size
            window = waveform[start_idx:end_idx]
            rms = float(np.sqrt(np.mean(window ** 2)))
            speech_active = rms >= self.vad_thresh

            # Slice frames directly from precomputed full-file spectrogram
            f_start = int(round(start_idx / HOP_LENGTH))
            f_end = f_start + self.window_frames
            if f_end <= total_spec_frames:
                spec_slice = full_spec[:, f_start:f_end]
            else:
                # Pad frames if near right boundary
                spec_slice = full_spec[:, f_start:total_spec_frames]
                pad_frames = self.window_frames - spec_slice.shape[1]
                spec_slice = np.pad(spec_slice, ((0, 0), (0, pad_frames)), mode="edge")

            raw_meta.append((start_idx / SAMPLE_RATE, end_idx / SAMPLE_RATE))
            is_speech.append(speech_active)
            raw_spec_slices.append(spec_slice)

        # Include tail if significant unvisited audio remains (>0.25s)
        tail_samples = total_samples - (last_start + self.window_size)
        if tail_samples >= int(0.25 * SAMPLE_RATE):
            tail_wave = waveform[-self.window_size:]
            rms = float(np.sqrt(np.mean(tail_wave ** 2)))
            tail_speech = rms >= self.vad_thresh

            f_start = max(0, total_spec_frames - self.window_frames)
            spec_slice = full_spec[:, f_start:f_start + self.window_frames]
            if spec_slice.shape[1] < self.window_frames:
                pad_frames = self.window_frames - spec_slice.shape[1]
                spec_slice = np.pad(spec_slice, ((0, 0), (0, pad_frames)), mode="edge")

            tail_st = max(0.0, (total_samples - self.window_size) / SAMPLE_RATE)
            tail_et = total_samples / SAMPLE_RATE
            raw_meta.append((tail_st, tail_et))
            is_speech.append(tail_speech)
            raw_spec_slices.append(spec_slice)

        if not raw_meta:
            return []

        # Step 3: VAD Hangover Smoothing (Bridge brief pauses < 0.3s)
        smoothed_speech = list(is_speech)
        for i in range(1, len(is_speech) - 1):
            if not is_speech[i] and is_speech[i - 1] and is_speech[i + 1]:
                smoothed_speech[i] = True

        valid_specs = []
        window_meta = []
        for i, active in enumerate(smoothed_speech):
            if active:
                valid_specs.append(raw_spec_slices[i])
                window_meta.append(raw_meta[i])

        if not valid_specs:
            return []

        # Step 4: Batched Neural Embedding Extraction from Pre-Sliced Spectrograms
        X = self.identifier.extract_batch_embeddings_from_specs(valid_specs, batch_size=32)

        # Step 5: Optional dimensionality reduction for clustering
        X_clust = X
        if len(X) > 50 and X.shape[1] > 16:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=16, random_state=42)
            X_clust = svd.fit_transform(X)

        # Step 6: Determine Number of Speaker Clusters (Guarded against O(N^2) stalls)
        if num_speakers is None or num_speakers <= 0:
            if len(X) < 4:
                n_clusters = 1
            else:
                best_k = 2
                best_score = -1.0
                max_k = min(6, len(X))
                
                # Subsample if N > 100 to prevent silhouette O(N^2) stall
                if len(X_clust) > 100:
                    sub_idx = np.linspace(0, len(X_clust) - 1, 80, dtype=int)
                    eval_X = X_clust[sub_idx]
                else:
                    eval_X = X_clust

                for k in range(2, max_k):
                    try:
                        clu = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                    except TypeError:
                        clu = AgglomerativeClustering(n_clusters=k, linkage="average", **{"affinity": "cosine"})
                    labs = clu.fit_predict(eval_X)
                    if len(set(labs)) > 1:
                        score = float(silhouette_score(eval_X, labs, metric="cosine"))
                        if score > best_score:
                            best_score = score
                            best_k = k
                n_clusters = best_k if best_score > 0.15 else 1
        else:
            n_clusters = min(num_speakers, len(X))

        # Step 7: Cluster Window Embeddings
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

        # Step 8: Build Non-Overlapping Timeline Segments
        raw_segments = []
        for i, (st, et) in enumerate(window_meta):
            spk_label = f"Speaker_{labels[i]}"
            raw_segments.append(DiarizationSegment(st, et, spk_label, X[i]))

        merged_segments: List[DiarizationSegment] = []
        current = raw_segments[0]

        for next_seg in raw_segments[1:]:
            if next_seg.speaker == current.speaker:
                # Same speaker: extend segment
                current.end = max(current.end, next_seg.end)
            else:
                # Different speaker transition: resolve overlapping boundary to midpoint
                if next_seg.start < current.end:
                    boundary = round((current.end + next_seg.start) / 2.0, 2)
                    current.end = max(current.start + 0.1, boundary)
                    next_seg.start = current.end

                merged_segments.append(current)
                current = next_seg

        merged_segments.append(current)

        # Enforce chronological monotonic timeline invariant: turn[i].end <= turn[i+1].start
        for i in range(len(merged_segments) - 1):
            if merged_segments[i].end > merged_segments[i + 1].start:
                mid = round((merged_segments[i].end + merged_segments[i + 1].start) / 2.0, 2)
                merged_segments[i].end = mid
                merged_segments[i + 1].start = mid

        return merged_segments
