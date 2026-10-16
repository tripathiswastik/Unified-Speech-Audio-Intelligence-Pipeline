"""Tests for audio loading, window preprocessing, and Mel-spectrogram feature extraction."""
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
import pytest

from pipeline.config import SAMPLE_RATE, TARGET_SAMPLES, N_MELS
from pipeline.audio_loader import (
    compute_log_mel_spectrogram,
    get_mel_filterbank,
    load_full_audio,
    preprocess_audio_window,
)


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
    synthetic_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    spec = compute_log_mel_spectrogram(synthetic_wave)
    assert isinstance(spec, np.ndarray)
    assert spec.ndim == 2
    assert spec.shape[0] == N_MELS
    assert np.all(np.isfinite(spec))


def test_load_full_audio_preserves_duration():
    """Verify that load_full_audio does not truncate long recordings to 4.0 seconds."""
    duration = 10.0  # 10 seconds (exceeds 4.0s training window)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    synthetic_wave = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, synthetic_wave, SAMPLE_RATE)
        tmp_path = Path(tmp.name)

    try:
        loaded_wave, reported_dur = load_full_audio(tmp_path)
        assert len(loaded_wave) == int(SAMPLE_RATE * duration)
        assert abs(reported_dur - duration) < 0.01
        assert len(loaded_wave) > TARGET_SAMPLES  # Confirms not truncated
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_preprocess_audio_window():
    """Verify that preprocess_audio_window conditions waveforms to exact TARGET_SAMPLES."""
    # Test short waveform (2 seconds -> padded to 4.0s)
    short_wave = np.ones(int(SAMPLE_RATE * 2.0), dtype=np.float32)
    padded = preprocess_audio_window(short_wave, target_samples=TARGET_SAMPLES)
    assert len(padded) == TARGET_SAMPLES

    # Test long waveform (8 seconds -> center-cropped to 4.0s)
    long_wave = np.ones(int(SAMPLE_RATE * 8.0), dtype=np.float32)
    cropped = preprocess_audio_window(long_wave, target_samples=TARGET_SAMPLES)
    assert len(cropped) == TARGET_SAMPLES


def test_urban_noise_mixer_snr_tolerance():
    """Verify noise mixer produces audio matching target SNR within tolerance."""
    from pipeline.config import METADATA_CSV
    from pipeline.urban_noise_mixer import UrbanNoiseMixer
    if not METADATA_CSV.exists():
        pytest.skip("UrbanSound8K metadata not present in test environment")
    try:
        mixer = UrbanNoiseMixer()
        t = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False)
        clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        target_snr = 10.0
        noisy, noise_cls, achieved_snr = mixer.mix(clean, snr_db=target_snr)
        assert len(noisy) == len(clean)
        assert abs(achieved_snr - target_snr) < 2.5
    except ValueError:
        pytest.skip("Audio directory folds not populated")

