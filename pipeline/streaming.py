"""Real-Time Streaming Audio Intelligence Pipeline.
Includes:
1. StreamingPCEN: Auto-regressive Per-Channel Energy Normalization for rolling audio buffers.
2. StreamingDiarizer: Low-latency online sliding buffer diarization with voiceprint tracking,
   multi-hop chunk ingestion, and true turn segmentation with speaker-change smoothing.
3. StreamingSpeechEvent: Structured payload for window events and segmented speech turns.
"""
from typing import Dict, List, Tuple, Optional, Generator, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from pipeline.config import CHECKPOINTS_DIR, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH
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
    event_type: str = "window"  # "window", "turn_start", "turn_complete", "speaker_shift"


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
    """Low-latency online streaming diarizer operating over rolling audio buffers.
    
    Features:
    - Multi-hop chunk support: correctly processes every elapsed hop when chunks span multiple boundaries.
    - Trained checkpoint provenance: automatically loads trained speaker model by default.
    - True turn segmentation: aggregates consecutive speaker frames into speech turns with speaker-shift smoothing.
    """

    def __init__(
        self,
        embedding_net: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        buffer_duration: float = 2.0,
        hop_duration: float = 0.5,
        sr: int = SAMPLE_RATE,
        energy_threshold: float = 0.015,
        similarity_threshold: float = 0.65,
        smoothing_frames: int = 2,
        hangover_frames: int = 1,
    ):
        self.device = device if device is not None else get_device(verbose=False)
        self.sr = sr
        self.buffer_size = int(buffer_duration * sr)
        self.hop_size = int(hop_duration * sr)
        self.energy_threshold = energy_threshold
        self.similarity_threshold = similarity_threshold
        self.smoothing_frames = smoothing_frames
        self.hangover_frames = hangover_frames

        # Load verified trained speaker model if not explicitly provided
        if embedding_net is not None:
            self.embedding_net = embedding_net
        else:
            spk_chk = CHECKPOINTS_DIR / "best_speaker_model.pt"
            if spk_chk.exists():
                try:
                    state = torch.load(spk_chk, map_location=self.device)
                    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
                    has_se = any("fc" in k for k in state_dict.keys())
                    backbone = "se_resnet" if has_se else "baseline"
                    self.embedding_net = SpeakerEmbeddingNet(embedding_dim=128, backbone=backbone)
                    self.embedding_net.load_state_dict(state_dict)
                except Exception:
                    self.embedding_net = SESpeakerEmbeddingNet()
            else:
                self.embedding_net = SESpeakerEmbeddingNet()
        self.embedding_net.to(self.device).eval()

        # Rolling audio stream buffer
        self.audio_history = np.zeros(0, dtype=np.float32)
        self.total_samples_ingested = 0

        # Online speaker profile dictionary: speaker_id -> running_mean_embedding (128-d)
        self.active_speakers: Dict[str, np.ndarray] = {}
        self.speaker_counts: Dict[str, int] = {}
        self.next_speaker_idx = 1

        # Turn segmentation and speaker-change smoothing state
        self.current_turn_speaker: Optional[str] = None
        self.current_turn_start: float = 0.0
        self.current_turn_end: float = 0.0
        self.current_turn_confidences: List[float] = []
        self.pending_speaker: Optional[str] = None
        self.pending_speaker_count: int = 0
        self.silence_frame_count: int = 0

    def reset(self):
        """Clear streaming state, enrolled speaker history, and turn tracking."""
        self.audio_history = np.zeros(0, dtype=np.float32)
        self.total_samples_ingested = 0
        self.active_speakers.clear()
        self.speaker_counts.clear()
        self.next_speaker_idx = 1
        self.current_turn_speaker = None
        self.current_turn_start = 0.0
        self.current_turn_end = 0.0
        self.current_turn_confidences.clear()
        self.pending_speaker = None
        self.pending_speaker_count = 0
        self.silence_frame_count = 0

    def push_audio(self, pcm_chunk: np.ndarray) -> List[StreamingSpeechEvent]:
        """Ingest a chunk of raw PCM audio and emit events for EVERY elapsed hop boundary."""
        events: List[StreamingSpeechEvent] = []
        if len(pcm_chunk) == 0:
            return events

        prev_ingested = self.total_samples_ingested
        self.total_samples_ingested += len(pcm_chunk)
        self.audio_history = np.concatenate([self.audio_history, pcm_chunk.astype(np.float32)])

        # Trigger inference for every hop boundary that has elapsed
        hops_before = prev_ingested // self.hop_size
        hops_after = self.total_samples_ingested // self.hop_size

        for h in range(hops_before + 1, hops_after + 1):
            h_sample_end = h * self.hop_size
            if h_sample_end < self.buffer_size:
                continue

            # Calculate slice in audio_history
            history_start_sample = self.total_samples_ingested - len(self.audio_history)
            win_end_idx = h_sample_end - history_start_sample
            win_start_idx = win_end_idx - self.buffer_size

            if win_start_idx >= 0 and win_end_idx <= len(self.audio_history):
                window_audio = self.audio_history[win_start_idx:win_end_idx]
                t_end = round(h_sample_end / self.sr, 2)
                t_start = round(max(0.0, (h_sample_end - self.buffer_size) / self.sr), 2)
                window_event = self._process_window(window_audio, t_start=t_start, t_end=t_end)
                if window_event is not None:
                    turn_events = self._update_turn_tracking(window_event)
                    events.append(window_event)
                    events.extend(turn_events)

        # Trim audio_history so it only retains what could be needed for future hops (buffer_size * 2)
        if len(self.audio_history) > self.buffer_size * 2:
            self.audio_history = self.audio_history[-self.buffer_size * 2:]

        return events

    def _update_turn_tracking(self, event: StreamingSpeechEvent) -> List[StreamingSpeechEvent]:
        """Track ongoing speech turns, apply speaker change smoothing, and emit finalized turns."""
        emitted: List[StreamingSpeechEvent] = []
        if not event.is_speech:
            self.silence_frame_count += 1
            if self.silence_frame_count > self.hangover_frames and self.current_turn_speaker is not None:
                # Finalize turn on silence
                turn_ev = StreamingSpeechEvent(
                    timestamp_start=self.current_turn_start,
                    timestamp_end=self.current_turn_end,
                    speaker_tag=self.current_turn_speaker,
                    confidence=round(float(np.mean(self.current_turn_confidences)), 3) if self.current_turn_confidences else 1.0,
                    is_speech=True,
                    embedding=event.embedding,
                    event_type="turn_complete",
                )
                emitted.append(turn_ev)
                self.current_turn_speaker = None
                self.current_turn_confidences = []
                self.pending_speaker = None
                self.pending_speaker_count = 0
            return emitted

        # Speech detected
        self.silence_frame_count = 0
        detected_spk = event.speaker_tag

        if self.current_turn_speaker is None:
            # Start new turn
            self.current_turn_speaker = detected_spk
            self.current_turn_start = event.timestamp_start
            self.current_turn_end = event.timestamp_end
            self.current_turn_confidences = [event.confidence]
            self.pending_speaker = None
            self.pending_speaker_count = 0
            emitted.append(StreamingSpeechEvent(
                timestamp_start=self.current_turn_start,
                timestamp_end=self.current_turn_end,
                speaker_tag=self.current_turn_speaker,
                confidence=event.confidence,
                is_speech=True,
                embedding=event.embedding,
                event_type="turn_start",
            ))
        elif detected_spk == self.current_turn_speaker:
            # Extend existing turn
            self.current_turn_end = max(self.current_turn_end, event.timestamp_end)
            self.current_turn_confidences.append(event.confidence)
            self.pending_speaker = None
            self.pending_speaker_count = 0
        else:
            # Speaker change candidate - apply smoothing
            if self.pending_speaker == detected_spk:
                self.pending_speaker_count += 1
            else:
                self.pending_speaker = detected_spk
                self.pending_speaker_count = 1

            if self.pending_speaker_count >= self.smoothing_frames:
                # Confirmed speaker shift: complete previous turn and start new turn
                turn_ev = StreamingSpeechEvent(
                    timestamp_start=self.current_turn_start,
                    timestamp_end=self.current_turn_end,
                    speaker_tag=self.current_turn_speaker,
                    confidence=round(float(np.mean(self.current_turn_confidences)), 3) if self.current_turn_confidences else 1.0,
                    is_speech=True,
                    embedding=event.embedding,
                    event_type="turn_complete",
                )
                emitted.append(turn_ev)

                # Announce speaker shift
                shift_ev = StreamingSpeechEvent(
                    timestamp_start=event.timestamp_start,
                    timestamp_end=event.timestamp_end,
                    speaker_tag=detected_spk,
                    confidence=event.confidence,
                    is_speech=True,
                    embedding=event.embedding,
                    event_type="speaker_shift",
                )
                emitted.append(shift_ev)

                # Initialize new turn
                self.current_turn_speaker = detected_spk
                self.current_turn_start = event.timestamp_start
                self.current_turn_end = event.timestamp_end
                self.current_turn_confidences = [event.confidence]
                self.pending_speaker = None
                self.pending_speaker_count = 0
            else:
                # Still within smoothing window, extend previous turn end time slightly
                self.current_turn_end = max(self.current_turn_end, event.timestamp_end)

        return emitted

    def finalize_stream(self) -> List[StreamingSpeechEvent]:
        """Flush and finalize any in-progress speech turn when stream closes."""
        emitted: List[StreamingSpeechEvent] = []
        if self.current_turn_speaker is not None:
            turn_ev = StreamingSpeechEvent(
                timestamp_start=self.current_turn_start,
                timestamp_end=self.current_turn_end,
                speaker_tag=self.current_turn_speaker,
                confidence=round(float(np.mean(self.current_turn_confidences)), 3) if self.current_turn_confidences else 1.0,
                is_speech=True,
                embedding=np.zeros(128, dtype=np.float32),
                event_type="turn_complete",
            )
            emitted.append(turn_ev)
            self.current_turn_speaker = None
            self.current_turn_confidences = []
        return emitted

    def _process_window(
        self, window_audio: np.ndarray, t_start: Optional[float] = None, t_end: Optional[float] = None
    ) -> Optional[StreamingSpeechEvent]:
        """Extract voiceprint and track online speaker assignment for the current window."""
        if t_end is None:
            t_end = round(self.total_samples_ingested / self.sr, 2)
        if t_start is None:
            t_start = round(max(0.0, t_end - (self.buffer_size / self.sr)), 2)

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
                event_type="window",
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
            timestamp_start=t_start,
            timestamp_end=t_end,
            speaker_tag=assigned_spk,
            confidence=round(conf, 3),
            is_speech=True,
            embedding=emb,
            event_type="window",
        )
