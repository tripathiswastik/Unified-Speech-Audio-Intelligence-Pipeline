"""Audio Loader & Preprocessor for UrbanSound8K and Continuous Speech Pipelines.
Handles 24-bit PCM, WAVE_FORMAT_EXTENSIBLE, multi-channel, and variable sample rates.
Provides both full-length audio loading for diarization/conversations and fixed-window
preprocessing for convolutional classification networks.
"""
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import numpy as np
import soundfile as sf
import librosa

from pipeline.config import SAMPLE_RATE, TARGET_SAMPLES, N_MELS, N_FFT, HOP_LENGTH


def load_full_audio(
    file_path: Path, target_sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, float]:
    """Load an audio file in its entirety, convert to mono, resample, and normalize.
    
    Does NOT truncate or pad to fixed length, preserving full continuous duration
    and absolute timestamps for multi-speaker diarization and conversational analysis.

    Returns:
        Tuple of (waveform: np.ndarray [float32], duration_seconds: float)
    """
    file_str = str(file_path)
    try:
        # Fast C-level reading with soundfile (supports 24-bit, extensible PCM)
        waveform, sr = sf.read(file_str, dtype="float32")
    except Exception:
        # Fallback to librosa if soundfile encounters an atypical container/codec
        waveform, sr = librosa.load(file_str, sr=None, mono=False)

    # Convert to mono if multi-channel (samples, channels) or (channels, samples)
    if waveform.ndim > 1:
        if waveform.shape[0] < waveform.shape[1] and waveform.shape[0] <= 4:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)

    # Resample if sample rate doesn't match target_sr
    if sr != target_sr:
        try:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr, res_type="soxr_fast")
        except Exception:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Normalize amplitude to [-1.0, 1.0]
    max_val = np.max(np.abs(waveform)) if len(waveform) > 0 else 0.0
    if max_val > 1e-6:
        waveform = waveform / max_val

    duration = float(len(waveform)) / float(target_sr)
    return waveform.astype(np.float32), duration


def preprocess_audio_window(
    waveform: np.ndarray, target_samples: int = TARGET_SAMPLES
) -> np.ndarray:
    """Pad or center-crop a 1D waveform to an exact target number of samples.
    
    Used strictly by fixed-input convolutional neural networks (e.g., UrbanSound8K
    or fixed-window SER networks).
    """
    num_samples = len(waveform)
    if num_samples < target_samples:
        pad_width = target_samples - num_samples
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    elif num_samples > target_samples:
        start = (num_samples - target_samples) // 2
        waveform = waveform[start : start + target_samples]

    max_val = np.max(np.abs(waveform)) if len(waveform) > 0 else 0.0
    if max_val > 1e-6:
        waveform = waveform / max_val

    return waveform.astype(np.float32)


def load_and_preprocess_audio(
    file_path: Path,
    target_sr: int = SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Load an audio file and format to fixed length (TARGET_SAMPLES).
    
    Provided for training pipelines and fixed-input evaluations.
    """
    waveform, _ = load_full_audio(file_path, target_sr=target_sr)
    return preprocess_audio_window(waveform, target_samples=target_samples)


_MEL_BASIS_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def get_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Retrieve or compute a cached static triangular Mel filterbank matrix."""
    key = (sr, n_fft, n_mels)
    if key not in _MEL_BASIS_CACHE:
        _MEL_BASIS_CACHE[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return _MEL_BASIS_CACHE[key]


def compute_log_mel_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract Log-Mel Spectrogram with reused Mel filterbank and standard normalization."""
    mel_basis = get_mel_filterbank(sr, N_FFT, N_MELS)
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    power = np.abs(stft) ** 2
    mel_spec = np.dot(mel_basis, power)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    if std > 1e-6:
        log_mel = (log_mel - mean) / std
    else:
        log_mel = log_mel - mean
    return log_mel.astype(np.float32)


def compute_cmvn_log_mel_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract Log-Mel Spectrogram with Per-Channel Cepstral Mean & Variance Normalization (CMVN).
    
    Normalizes each Mel frequency channel independently across time, effectively suppressing
    stationary background environmental noise and channel frequency coloration.
    """
    mel_basis = get_mel_filterbank(sr, N_FFT, N_MELS)
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    power = np.abs(stft) ** 2
    mel_spec = np.dot(mel_basis, power)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    channel_mean = np.mean(log_mel, axis=1, keepdims=True)
    channel_std = np.std(log_mel, axis=1, keepdims=True)
    channel_std = np.where(channel_std > 1e-6, channel_std, 1.0)
    norm_mel = (log_mel - channel_mean) / channel_std
    return norm_mel.astype(np.float32)


def extract_ssl_features(
    waveform: np.ndarray,
    model_name: str = "facebook/wav2vec2-base",
    device: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Extract Self-Supervised Learning (SSL) representations (wav2vec 2.0 / HuBERT).

    Loads a pretrained SSL transformer from Hugging Face, freezes its parameters,
    and extracts frame-level hidden state features. Returns None if transformers is unavailable.
    """
    try:
        import importlib
        transformers_mod = importlib.import_module("transformers")
        AutoModel = getattr(transformers_mod, "AutoModel")
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        tensor_wave = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_wave)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return hidden_states
    except Exception:
        return None


