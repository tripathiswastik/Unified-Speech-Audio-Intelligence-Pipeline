"""Tests for Real-Time Streaming and ONNX Edge Export modules."""
import os
from pathlib import Path
import pytest
import numpy as np
import torch

from pipeline.config import CHECKPOINTS_DIR, SAMPLE_RATE, N_MELS, CLASSES
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
    """Validate low-latency online sliding buffer diarization."""

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

    def test_multi_hop_chunk_processing(self):
        """Verify that a large chunk crossing multiple hop boundaries processes all hops without skipping."""
        model = SESpeakerEmbeddingNet()
        diarizer = StreamingDiarizer(
            embedding_net=model,
            buffer_duration=1.0,
            hop_duration=0.5,
            sr=16000,
            energy_threshold=0.01,
        )

        # Initial prime of 1.0s buffer (buffer_size = 16000)
        prime_chunk = np.zeros(16000, dtype=np.float32)
        _ = diarizer.push_audio(prime_chunk)

        # Now send a 1.5s chunk (24,000 samples) at once, which spans exactly 3 hops (at 1.5s, 2.0s, 2.5s)
        t = np.linspace(0, 1.5, 24000, endpoint=False)
        large_chunk = (0.4 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        events = diarizer.push_audio(large_chunk)

        # Must have processed multiple window events
        window_events = [e for e in events if e.event_type == "window"]
        assert len(window_events) >= 3, f"Expected at least 3 window events for 1.5s chunk, got {len(window_events)}"

        # Verify timestamps advance monotonically
        end_times = [e.timestamp_end for e in window_events]
        assert end_times == sorted(end_times)
        assert end_times[-1] >= 2.5

    def test_turn_segmentation_and_smoothing(self):
        """Verify turn_start and turn_complete segmentation behavior."""
        model = SESpeakerEmbeddingNet()
        diarizer = StreamingDiarizer(
            embedding_net=model,
            buffer_duration=1.0,
            hop_duration=0.5,
            sr=16000,
            energy_threshold=0.01,
            smoothing_frames=2,
            hangover_frames=1,
        )

        # 1. Feed speech
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        speech_audio = (0.5 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
        events = diarizer.push_audio(speech_audio)

        turn_starts = [e for e in events if e.event_type == "turn_start"]
        assert len(turn_starts) >= 1

        # 2. Feed silence to trigger turn_complete
        silence_audio = np.zeros(24000, dtype=np.float32)
        silence_events = diarizer.push_audio(silence_audio)
        turn_completes = [e for e in silence_events if e.event_type == "turn_complete"]
        assert len(turn_completes) >= 1
        assert turn_completes[0].timestamp_end >= turn_starts[0].timestamp_start


class TestONNXExportAndEdgeProfiles:
    """Validate ONNX export artifacts and edge hardware profiling."""

    def test_strict_checkpoint_failure(self):
        """Verify that strict_checkpoints=True fails closed when checkpoint is missing."""
        non_existent = Path("non_existent_model_checkpoint.pt")
        with pytest.raises(FileNotFoundError):
            export_se_audiocnn_to_onnx(checkpoint_path=non_existent, strict_checkpoints=True)

        with pytest.raises(FileNotFoundError):
            export_se_speaker_embedding_to_onnx(checkpoint_path=non_existent, strict_checkpoints=True)

    def test_onnx_files_exist_and_numerical_match(self):
        """Verify ONNX export with trained checkpoints and numerical equivalence with onnxruntime."""
        audio_onnx = CHECKPOINTS_DIR / "se_audiocnn.onnx"
        spk_onnx = CHECKPOINTS_DIR / "se_speaker_embedding.onnx"

        # Export models requiring trained checkpoints (unless ALLOW_UNTRAINED_ONNX=1 in CI) and verifying numerical equivalence
        allow_untrained = os.getenv("ALLOW_UNTRAINED_ONNX", "0").strip().lower() in ("1", "true", "yes")
        res_audio = export_se_audiocnn_to_onnx(strict_checkpoints=not allow_untrained, verify_equivalence=True)
        res_spk = export_se_speaker_embedding_to_onnx(strict_checkpoints=not allow_untrained, verify_equivalence=True)

        assert res_audio.exists(), f"Missing ONNX file: {res_audio}"
        assert res_spk.exists(), f"Missing ONNX file: {res_spk}"
        assert res_audio.stat().st_size > 50_000, f"SEAudioCNN ONNX size too small: {res_audio.stat().st_size} bytes"
        assert res_spk.stat().st_size > 50_000, f"SESpeakerEmbeddingNet ONNX size too small: {res_spk.stat().st_size} bytes"

    def test_edge_profiling_latency(self):
        profiles = profile_edge_frontends(num_runs=5)
        assert "SE_CNN_Edge" in profiles
        assert "SSL_Transformer_FrontEnd" in profiles

        se_prof = profiles["SE_CNN_Edge"]
        ssl_prof = profiles["SSL_Transformer_FrontEnd"]

        assert se_prof["model_size_mb"] < 10.0
        assert ssl_prof["model_size_mb"] > 200.0
        assert ssl_prof["is_estimate"] is True
        assert se_prof["is_estimate"] is False
        assert se_prof["mean_latency_ms"] < ssl_prof["mean_latency_ms"]


class TestUrbanNoiseMixerAPI:
    """Validate noise injection integration."""

    def test_urban_noise_mix_noise_interface(self):
        from pipeline.urban_noise_mixer import UrbanNoiseMixer
        mixer = UrbanNoiseMixer(classes=["siren"])
        dummy_speech = np.random.uniform(-0.1, 0.1, size=16000).astype(np.float32)
        mixed, achieved_snr = mixer.mix_noise(dummy_speech, noise_class="siren", target_snr=10.0)

        assert len(mixed) == len(dummy_speech)
        assert np.all(np.isfinite(mixed))
        assert isinstance(achieved_snr, float)
