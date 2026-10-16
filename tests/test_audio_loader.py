"""Tests for audio preprocessing and Mel-spectrogram feature extraction."""
import numpy as np
import pytest
from pipeline.config import SAMPLE_RATE, N_MELS
from pipeline.audio_loader import compute_log_mel_spectrogram, get_mel_filterbank


def test_get_mel_filterbank_caching():
    """Verify that mel filterbank basis is cached and returns stable shape."""
    fb1 = get_mel_filterbank(sr=SAMPLE_RATE, n_fft=2048, n_mels=N_MELS)
    fb2 = get_mel_filterbank(sr=SAMPLE_RATE, n_fft=2048, n_mels=N_MELS)
    assert fb1 is fb2  # Identity check: cached memory pointer
    assert fb1.shape == (N_MELS, 1025)


def test_compute_log_mel_spectrogram_synthetic():
    """Verify log-mel spectrogram computation on synthetic 2-second sine wave."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Generate 440 Hz tone
    synthetic_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    spec = compute_log_mel_spectrogram(synthetic_wave)
    assert isinstance(spec, np.ndarray)
    assert spec.ndim == 2
    assert spec.shape[0] == N_MELS
    # Value range normalized to [0, 1]
    assert np.all(spec >= 0.0)
    assert np.all(spec <= 1.0)
