"""Real-Time Streaming Audio Intelligence Pipeline.
Includes:
1. StreamingPCEN: Auto-regressive Per-Channel Energy Normalization for rolling audio buffers.
2. StreamingDiarizer: Low-latency online sliding ring buffer diarization with voiceprint tracking.
3. StreamingAudioProcessor: Real-time multi-task streaming processor for live microphone/WebSocket ingestion.
"""
from typing import Dict, List, Tuple, Optional, Generator, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import torch

from pipeline.config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH
from pipeline.audio_loader import get_mel_filterbank
from pipeline.model import SpeakerEmbeddingNet, SESpeakerEmbeddingNet
from pipeline.device import get_device


class StreamingSpeechEvent(NamedTuple):
    timestamp_start: float
    timestamp_end: float
    speaker_tag: str
    confidence: float
    is_speech: bool
    embedding: np.ndarray


class StreamingPCEN:
    """Online Per-Channel Energy Normalization (PCEN) for streaming rolling audio buffers.
    
    Operates incrementally with an auto-regressive state filter:
      s_c[t] = (1 - s) * s_c[t-1] + s * E_c[t]
      PCEN_c[t] = (E_c[t] / (eps + s_c[t])^alpha + delta)^gamma - delta^gamma
      
    Eliminates global future-context dependency, enabling real-time microphone ingestion.
    """
    def __init__(
        self,
        n_mels: int = N_MELS,
        s: float = 0.025,
        alpha: float = 0.98,
        delta: float = 2.0,
        gamma: float = 0.5,
        eps: float = 1e-6,
    ):
        self.n_mels = n_mels
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.eps = eps
        # State vector representing smoothed per-channel energy
        self.state: Optional[np.ndarray] = None

    def reset(self):
        """Reset historical smoother state for a new audio stream."""
        self.state = None

    def process_frame(self, energy_frame: np.ndarray) -> np.ndarray:
        """Process a single (n_mels,) frame or (n_mels, T) chunk incrementally."""
        if energy_frame.ndim == 1:
            energy_frame = energy_frame[:, np.newaxis]

        n_mels, t_steps = energy_frame.shape
        out = np.zeros_like(energy_frame, dtype=np.float32)

        if self.state is None:
            self.state = energy_frame[:, 0].copy()

        for t in range(t_steps):
            e_t = energy_frame[:, t]
            # Exponential moving average filter
            self.state = (1.0 - self.s) * self.state + self.s * e_t
            # Adaptive AGC compression
            smooth = (self.eps + self.state) ** self.alpha
            norm_e = (e_t / smooth) + self.delta
            out[:, t] = (norm_e ** self.gamma) - (self.delta ** self.gamma)

        return out if t_steps > 1 else out[:, 0]


class StreamingDiarizer:
    """Low-latency online streaming diarizer operating over rolling FIFO audio buffers."""

    def __init__(
        self,
        embedding_net: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        buffer_duration: float = 2.0,
        hop_duration: float = 0.5,
        sr: int = SAMPLE_RATE,
        energy_threshold: float = 0.015,
        similarity_threshold: float = 0.65,
    ):
        self.device = device if device is not None else get_device(verbose=False)
        self.sr = sr
        self.buffer_size = int(buffer_duration * sr)
        self.hop_size = int(hop_duration * sr)
        self.energy_threshold = energy_threshold
        self.similarity_threshold = similarity_threshold

        self.embedding_net = embedding_net if embedding_net is not None else SESpeakerEmbeddingNet()
        self.embedding_net.to(self.device).eval()

        # Rolling circular FIFO buffer
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.total_samples_ingested = 0

        # Online speaker profile dictionary: speaker_id -> running_mean_embedding (128-d)
        self.active_speakers: Dict[str, np.ndarray] = {}
        self.speaker_counts: Dict[str, int] = {}
        self.next_speaker_idx = 1

    def reset(self):
        """Clear streaming state and enrolled speaker history."""
        self.ring_buffer.fill(0.0)
        self.total_samples_ingested = 0
        self.active_speakers.clear()
        self.speaker_counts.clear()
        self.next_speaker_idx = 1

    def push_audio(self, pcm_chunk: np.ndarray) -> List[StreamingSpeechEvent]:
        """Ingest a chunk of raw PCM audio and emit detected speaker events when a hop boundary passes."""
        events: List[StreamingSpeechEvent] = []
        if len(pcm_chunk) == 0:
            return events

        # Append new samples into FIFO ring buffer
        num_new = len(pcm_chunk)
        if num_new >= self.buffer_size:
            self.ring_buffer = pcm_chunk[-self.buffer_size:].astype(np.float32)
        else:
            self.ring_buffer = np.roll(self.ring_buffer, -num_new)
            self.ring_buffer[-num_new:] = pcm_chunk.astype(np.float32)

        prev_ingested = self.total_samples_ingested
        self.total_samples_ingested += num_new

        # Trigger inference whenever a hop size has elapsed
        if self.total_samples_ingested >= self.buffer_size:
            hops_before = prev_ingested // self.hop_size
            hops_after = self.total_samples_ingested // self.hop_size

            if hops_after > hops_before:
                window_audio = self.ring_buffer.copy()
                event = self._process_window(window_audio)
                if event is not None:
                    events.append(event)

        return events

    def _process_window(self, window_audio: np.ndarray) -> Optional[StreamingSpeechEvent]:
        """Extract voiceprint and track online speaker assignment for the current window."""
        t_end = self.total_samples_ingested / self.sr
        t_start = max(0.0, t_end - (self.buffer_size / self.sr))

        # Fast VAD check using RMS energy
        rms = float(np.sqrt(np.mean(window_audio ** 2)))
        if rms < self.energy_threshold:
            return StreamingSpeechEvent(
                timestamp_start=t_start,
                timestamp_end=t_end,
                speaker_tag="Silence",
                confidence=1.0 - (rms / max(self.energy_threshold, 1e-6)),
                is_speech=False,
                embedding=np.zeros(128, dtype=np.float32),
            )

        # Mel-spectrogram feature extraction
        mel_basis = get_mel_filterbank(self.sr, N_FFT, N_MELS)
        import librosa
        stft = librosa.stft(window_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        power = np.abs(stft) ** 2
        mel_spec = np.dot(mel_basis, power)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        mean, std = np.mean(log_mel), np.std(log_mel)
        norm_mel = (log_mel - mean) / (std if std > 1e-6 else 1.0)

        tensor = torch.from_numpy(norm_mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.embedding_net(tensor)[0].cpu().numpy()

        # Online speaker assignment using cosine similarity
        best_spk: Optional[str] = None
        best_sim = -1.0

        for spk_name, ref_emb in self.active_speakers.items():
            sim = float(np.dot(emb, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_spk = spk_name

        if best_spk is None or best_sim < self.similarity_threshold:
            # Assign new online speaker cluster
            new_spk = f"Speaker_{self.next_speaker_idx}"
            self.next_speaker_idx += 1
            self.active_speakers[new_spk] = emb
            self.speaker_counts[new_spk] = 1
            assigned_spk = new_spk
            conf = 1.0
        else:
            # Update running speaker representation with EMA
            cnt = self.speaker_counts[best_spk]
            updated = (self.active_speakers[best_spk] * cnt + emb) / (cnt + 1)
            updated = updated / max(np.linalg.norm(updated), 1e-12)
            self.active_speakers[best_spk] = updated
            self.speaker_counts[best_spk] = min(cnt + 1, 50)
            assigned_spk = best_spk
            conf = best_sim

        return StreamingSpeechEvent(
            timestamp_start=round(t_start, 2),
            timestamp_end=round(t_end, 2),
            speaker_tag=assigned_spk,
            confidence=round(conf, 3),
            is_speech=True,
            embedding=emb,
        )
