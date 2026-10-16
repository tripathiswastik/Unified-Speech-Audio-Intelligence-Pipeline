"""Tests for SpeakerIdentifier and batched SpeakerDiarizer."""
import numpy as np
import torch
import pytest
from pipeline.config import SAMPLE_RATE
from pipeline.speaker_id import SpeakerEmbeddingNet, SpeakerIdentifier
from pipeline.diarization import SpeakerDiarizer


def test_speaker_embedding_net_shapes():
    """Verify forward pass output shape and L2 normalization."""
    net = SpeakerEmbeddingNet()
    net.eval()
    dummy_input = torch.randn(4, 1, 128, 173)
    with torch.no_grad():
        emb = net(dummy_input)
    assert emb.shape == (4, 128)
    # Norms must equal 1.0 (L2 normalized)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_speaker_identifier_vectorized():
    """Verify enrollment, matrix update, and vectorized identification."""
    identifier = SpeakerIdentifier()
    # Create two distinct dummy profiles directly
    emb1 = np.zeros(128, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(128, dtype=np.float32)
    emb2[1] = 1.0

    identifier.enrolled_profiles["Alice"] = emb1
    identifier.enrolled_profiles["Bob"] = emb2
    identifier._update_enrolled_matrix()

    assert identifier._enrolled_matrix is not None
    assert identifier._enrolled_matrix.shape == (2, 128)

    # Synthetic waveform
    wave = np.random.randn(int(SAMPLE_RATE * 1.5)).astype(np.float32)
    best_spk, sim = identifier.identify(wave, threshold=0.1)
    assert best_spk in ["Alice", "Bob"] or "Unenrolled" in best_spk
    assert isinstance(sim, float)


def test_batched_diarization():
    """Verify diarization executes batched window forward passes on synthetic audio."""
    diarizer = SpeakerDiarizer(window_size_sec=1.0, hop_size_sec=0.5, vad_energy_threshold=0.001)
    # 4 seconds of synthetic audio
    wave = np.random.randn(int(SAMPLE_RATE * 4.0)).astype(np.float32)
    segments = diarizer.diarize(wave, num_speakers=2)
    assert isinstance(segments, list)
    if segments:
        for s in segments:
            assert s.start < s.end
            assert s.speaker.startswith("Speaker_")
            assert s.embedding.shape == (128,)


def test_auto_speaker_count_diarization():
    """Verify diarizer can automatically estimate speaker count when num_speakers=None."""
    diarizer = SpeakerDiarizer(window_size_sec=1.0, hop_size_sec=0.5, vad_energy_threshold=0.001)
    # 6 seconds of synthetic audio
    wave = np.random.randn(int(SAMPLE_RATE * 6.0)).astype(np.float32)
    segments = diarizer.diarize(wave, num_speakers=None)
    assert isinstance(segments, list)


def test_diarization_timeline_monotonic_non_overlapping():
    """Verify strictly non-overlapping timeline invariant: turn[i].end <= turn[i+1].start."""
    diarizer = SpeakerDiarizer(window_size_sec=1.5, hop_size_sec=0.5, vad_energy_threshold=0.001)
    # 8 seconds of alternating synthetic speaker tones
    t = np.linspace(0, 8.0, int(SAMPLE_RATE * 8.0), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
    wave = wave.astype(np.float32)

    segments = diarizer.diarize(wave, num_speakers=2)
    assert len(segments) >= 1
    for i in range(len(segments) - 1):
        assert segments[i].end <= segments[i + 1].start + 1e-5, (
            f"Overlapping turns detected: Segment {i} ends at {segments[i].end}s "
            f"but Segment {i+1} starts at {segments[i+1].start}s"
        )


def test_diarization_short_audio_handling():
    """Verify short audio (0.8s, below 1.5s window) returns a valid segment rather than empty list."""
    diarizer = SpeakerDiarizer(window_size_sec=1.5, hop_size_sec=0.5, vad_energy_threshold=0.001)
    short_wave = (0.4 * np.random.randn(int(SAMPLE_RATE * 0.8))).astype(np.float32)
    segments = diarizer.diarize(short_wave, num_speakers=1)
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert abs(segments[0].end - 0.8) < 0.05


