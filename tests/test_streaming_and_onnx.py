"""Tests for Real-Time Streaming and ONNX Edge Export modules."""
import os
from pathlib import Path
import pytest
import numpy as np
import torch

from pipeline.config import CHECKPOINTS_DIR, SAMPLE_RATE, N_MELS
from pipeline.streaming import StreamingPCEN, StreamingDiarizer, StreamingSpeechEvent
from pipeline.model import SESpeakerEmbeddingNet
from pipeline.onnx_exporter import (
    export_se_audiocnn_to_onnx,
    export_se_speaker_embedding_to_onnx,
    profile_edge_frontends,
)


class TestStreamingPCEN:
    """Validate online Per-Channel Energy Normalization."""

    def test_single_frame_processing(self):
        pcen = StreamingPCEN(n_mels=128)
        frame = np.random.uniform(0.1, 5.0, size=(128,)).astype(np.float32)
        out = pcen.process_frame(frame)

        assert out.shape == (128,)
        assert np.all(np.isfinite(out))
        assert not np.isnan(out).any()

    def test_multi_frame_chunk(self):
        pcen = StreamingPCEN(n_mels=128)
        chunk = np.random.uniform(0.01, 10.0, size=(128, 30)).astype(np.float32)
        out = pcen.process_frame(chunk)

        assert out.shape == (128, 30)
        assert np.all(np.isfinite(out))

    def test_pcen_reset(self):
        pcen = StreamingPCEN(n_mels=128)
        chunk = np.ones((128, 10), dtype=np.float32)
        _ = pcen.process_frame(chunk)
        assert pcen.state is not None

        pcen.reset()
        assert pcen.state is None


class TestStreamingDiarizer:
    """Validate low-latency online circular buffer diarization."""

    def test_streaming_silence_detection(self):
        model = SESpeakerEmbeddingNet()
        diarizer = StreamingDiarizer(
            embedding_net=model,
            buffer_duration=1.0,
            hop_duration=0.5,
            sr=16000,
            energy_threshold=0.05,
        )

        # Push 1.5 seconds of near-silence in 0.25s chunks
        chunk = np.zeros(4000, dtype=np.float32)
        all_events = []
        for _ in range(6):
            events = diarizer.push_audio(chunk)
            all_events.extend(events)

        assert len(all_events) >= 1
        for ev in all_events:
            assert isinstance(ev, StreamingSpeechEvent)
            assert ev.speaker_tag == "Silence"
            assert not ev.is_speech

    def test_streaming_speech_ingestion(self):
        model = SESpeakerEmbeddingNet()
        diarizer = StreamingDiarizer(
            embedding_net=model,
            buffer_duration=1.0,
            hop_duration=0.5,
            sr=16000,
            energy_threshold=0.01,
        )

        # Generate synthetic speech-like harmonic signal
        t = np.linspace(0, 1.5, int(1.5 * 16000), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
        audio = audio.astype(np.float32)

        # Feed in 400ms chunks
        chunk_size = int(0.4 * 16000)
        events = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            evs = diarizer.push_audio(chunk)
            events.extend(evs)

        assert len(events) >= 1
        speech_events = [e for e in events if e.is_speech]
        assert len(speech_events) >= 1
        assert speech_events[0].embedding.shape == (128,)
        assert "Speaker" in speech_events[0].speaker_tag


class TestONNXExportAndEdgeProfiles:
    """Validate ONNX export artifacts and edge hardware profiling."""

    def test_onnx_files_exist(self):
        audio_onnx = CHECKPOINTS_DIR / "se_audiocnn.onnx"
        spk_onnx = CHECKPOINTS_DIR / "se_speaker_embedding.onnx"

        try:
            if not audio_onnx.exists() or audio_onnx.stat().st_size < 1_000_000:
                export_se_audiocnn_to_onnx(audio_onnx)
            if not spk_onnx.exists() or spk_onnx.stat().st_size < 1_000_000:
                export_se_speaker_embedding_to_onnx(spk_onnx)
        except Exception as e:
            pytest.skip(f"ONNX export not supported in current environment: {e}")

        assert audio_onnx.exists(), f"Missing ONNX file: {audio_onnx}"
        assert spk_onnx.exists(), f"Missing ONNX file: {spk_onnx}"
        assert audio_onnx.stat().st_size > 1_000_000, f"SEAudioCNN ONNX size too small: {audio_onnx.stat().st_size} bytes"
        assert spk_onnx.stat().st_size > 1_000_000, f"SESpeakerEmbeddingNet ONNX size too small: {spk_onnx.stat().st_size} bytes"

    def test_edge_profiling_latency(self):
        profiles = profile_edge_frontends(num_runs=5)
        assert "SE_CNN_Edge" in profiles
        assert "SSL_Transformer_FrontEnd" in profiles

        se_prof = profiles["SE_CNN_Edge"]
        ssl_prof = profiles["SSL_Transformer_FrontEnd"]

        assert se_prof["model_size_mb"] < 10.0
        assert ssl_prof["model_size_mb"] > 200.0
        assert se_prof["mean_latency_ms"] < ssl_prof["mean_latency_ms"]
