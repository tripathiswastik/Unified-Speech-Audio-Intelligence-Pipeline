"""Audio Loader & Preprocessor for UrbanSound8K.
Handles 24-bit PCM, WAVE_FORMAT_EXTENSIBLE, multi-channel, and variable sample rates.
"""
from pathlib import Path
from typing import Tuple, Dict
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

    # Pre-crop if duration exceeds 4.0s before resampling to save CPU cycles
    max_raw_samples = int(4.0 * sr)
    if len(waveform) > max_raw_samples:
        start = (len(waveform) - max_raw_samples) // 2
        waveform = waveform[start : start + max_raw_samples]

    # Resample if sample rate doesn't match target using high-speed soxr_fast
    if sr != target_sr:
        try:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr, res_type="soxr_fast")
        except Exception:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Standardize length to exactly TARGET_SAMPLES (4.0s at target_sr)
    num_samples = len(waveform)
    if num_samples < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - num_samples
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    elif num_samples > TARGET_SAMPLES:
        start = (num_samples - TARGET_SAMPLES) // 2
        waveform = waveform[start : start + TARGET_SAMPLES]

    # Normalize amplitude to [-1.0, 1.0]
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-6:
        waveform = waveform / max_val

    return waveform.astype(np.float32)


_MEL_BASIS_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def get_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Retrieve or precompute cached Mel filterbank basis to avoid per-call reallocations."""
    key = (sr, n_fft, n_mels)
    if key not in _MEL_BASIS_CACHE:
        _MEL_BASIS_CACHE[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return _MEL_BASIS_CACHE[key]


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Convert standardized waveform to a normalized Log-Mel Spectrogram using precomputed filterbanks."""
    mel_basis = get_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    power_spec = np.abs(stft) ** 2
    mel_spec = np.dot(mel_basis, power_spec)
    # Convert to power-to-db (Log-Mel)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] range for stable neural network training
    min_db = -80.0
    norm_mel = (np.clip(log_mel, min_db, 0.0) - min_db) / (-min_db)
    return norm_mel.astype(np.float32)
