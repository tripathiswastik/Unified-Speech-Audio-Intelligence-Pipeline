"""Audio Loader & Preprocessor for UrbanSound8K.
Handles 24-bit PCM, WAVE_FORMAT_EXTENSIBLE, multi-channel, and variable sample rates.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa

from pipeline.config import SAMPLE_RATE, TARGET_SAMPLES, N_MELS, N_FFT, HOP_LENGTH


def load_and_preprocess_audio(file_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, convert to mono, resample, and pad/crop to fixed length."""
    file_str = str(file_path)
    try:
        # soundfile is preferred for fast C-level reading and 24-bit/extensible format support
        waveform, sr = sf.read(file_str, dtype="float32")
    except Exception:
        # Fallback to librosa if soundfile encounters an atypical header
        waveform, sr = librosa.load(file_str, sr=None, mono=False)

    # Convert to mono if multi-channel (samples, channels) or (channels, samples)
    if waveform.ndim > 1:
        if waveform.shape[0] < waveform.shape[1] and waveform.shape[0] <= 4:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)

    # Resample if sample rate doesn't match target
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Standardize length to exactly TARGET_SAMPLES (4.0s)
    num_samples = len(waveform)
    if num_samples < TARGET_SAMPLES:
        # Pad with zeros (or repetition)
        pad_width = TARGET_SAMPLES - num_samples
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    elif num_samples > TARGET_SAMPLES:
        # Center-crop
        start = (num_samples - TARGET_SAMPLES) // 2
        waveform = waveform[start : start + TARGET_SAMPLES]

    # Normalize amplitude to [-1.0, 1.0]
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-6:
        waveform = waveform / max_val

    return waveform.astype(np.float32)


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Convert standardized waveform to a normalized Log-Mel Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    # Convert to power-to-db (Log-Mel)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] range for stable neural network training
    min_db = -80.0
    norm_mel = (np.clip(log_mel, min_db, 0.0) - min_db) / (-min_db)
    return norm_mel.astype(np.float32)
