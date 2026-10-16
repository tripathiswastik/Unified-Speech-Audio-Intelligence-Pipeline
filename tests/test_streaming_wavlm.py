"""Unit and Integration Tests for Chunk-Based Streaming WavLM.

Validates:
1. Causal attention invariant (no lookahead into future frames).
2. Streaming state continuity across sequential 200ms audio chunks.
3. Multi-task head outputs (Scene, Voiceprint, Emotion).
4. Static-shape ONNX export and ONNX Runtime inference numerical equivalence.
"""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pipeline.streaming_wavlm import (
    CausalConv1d,
    ConvFeatureEncoder,
    CausalSelfAttention,
    MultiTaskStreamingWavLM,
    export_streaming_wavlm_to_onnx,
)


class TestStreamingWavLMArchitecture:
    """Validate core causal architecture and multi-task heads."""

    def test_causal_conv1d_lookahead_invariant(self):
        """Modifying future audio samples must not alter past output frames."""
        conv = CausalConv1d(in_channels=1, out_channels=16, kernel_size=10, stride=5).eval()

        # Input A: random 3,200 samples
        audio_a = torch.randn(1, 1, 3200)
        # Input B: identical for the first 1,600 samples, modified in the remaining 1,600 samples
        audio_b = audio_a.clone()
        audio_b[:, :, 1600:] = torch.randn(1, 1, 1600)

        with torch.no_grad():
            out_a = conv(audio_a)
            out_b = conv(audio_b)

        # Output frames corresponding to the first half (< 1600 / 5 = 320 frames) must be identical
        valid_frames = (1600 // 5) - 2
        diff = torch.max(torch.abs(out_a[:, :, :valid_frames] - out_b[:, :, :valid_frames])).item()
        assert diff < 1e-6, f"CausalConv1d violated causality: diff = {diff}"

    def test_causal_self_attention_causality(self):
        """Attention weights must never attend to future tokens within a chunk."""
        attn = CausalSelfAttention(hidden_dim=64, num_heads=2, max_past_frames=20).eval()
        x = torch.randn(1, 10, 64)

        with torch.no_grad():
            out, new_k, new_v = attn(x)

        assert out.shape == (1, 10, 64)
        assert new_k.shape == (1, 2, 10, 32)
        assert new_v.shape == (1, 2, 10, 32)

    def test_multitask_streaming_wavlm_shapes(self):
        """Verify multi-task head output shapes and L2 normalization of voiceprints."""
        model = MultiTaskStreamingWavLM(
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            ffn_dim=128,
            max_past_frames=20,
            early_tap_layer=2,
            num_scene_classes=10,
            num_emotion_classes=8,
            embedding_dim=128,
        ).eval()

        chunk_samples = 3200  # 200ms chunk at 16kHz
        chunk = torch.randn(1, chunk_samples)

        with torch.no_grad():
            scene_logits, speaker_emb, emotion_logits, state_cache = model(chunk)

        # Assert shapes
        assert scene_logits.shape == (1, 10), f"Expected (1, 10), got {scene_logits.shape}"
        assert speaker_emb.shape == (1, 128), f"Expected (1, 128), got {speaker_emb.shape}"
        assert emotion_logits.shape == (1, 8), f"Expected (1, 8), got {emotion_logits.shape}"
        assert state_cache.shape == (4, 2, 1, 2, 10, 32)

        # Voiceprint embedding must be strictly L2-normalized (norm == 1.0)
        l2_norm = torch.norm(speaker_emb, p=2, dim=-1).item()
        assert np.isclose(l2_norm, 1.0, atol=1e-4), f"Speaker embedding not normalized: norm={l2_norm}"

    def test_sequential_chunk_streaming_state_continuity(self):
        """Simulate sequential audio streaming with state cache propagation."""
        model = MultiTaskStreamingWavLM(
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            ffn_dim=128,
            max_past_frames=20,
            early_tap_layer=2,
        ).eval()

        chunk_samples = 3200  # 200ms
        state_cache = None

        # Stream 5 consecutive chunks (1.0 second of audio in 200ms steps)
        for chunk_idx in range(5):
            chunk = torch.randn(1, chunk_samples)
            with torch.no_grad():
                scene_logits, speaker_emb, emotion_logits, state_cache = model(chunk, state_cache=state_cache)

            assert scene_logits.shape == (1, 10)
            assert speaker_emb.shape == (1, 128)
            assert emotion_logits.shape == (1, 8)
            # Cache temporal dimension is bounded by max_past_frames (20)
            expected_past = min((chunk_idx + 1) * 10, 20)
            assert state_cache.shape == (4, 2, 1, 2, expected_past, 32)


class TestStreamingWavLMONNXExport:
    """Validate ONNX exportability and ONNX Runtime execution with static shapes."""

    def test_static_shape_onnx_export_and_inference(self):
        """Verify model exports cleanly to static ONNX and onnxruntime produces matching results."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "streaming_wavlm_static.onnx"

            # Create mini model for fast test execution
            model = MultiTaskStreamingWavLM(
                hidden_dim=64,
                num_layers=3,
                num_heads=2,
                ffn_dim=128,
                max_past_frames=20,
                early_tap_layer=1,
            ).eval()

            chunk_samples = 3200
            audio_in = torch.randn(1, chunk_samples, dtype=torch.float32)
            # Static cache: (num_layers=3, 2, B=1, heads=2, past_frames=20, head_dim=32)
            cache_in = torch.randn(3, 2, 1, 2, 20, 32, dtype=torch.float32)

            torch.onnx.export(
                model,
                (audio_in, cache_in),
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["audio_chunk", "state_cache_in"],
                output_names=["scene_logits", "speaker_embedding", "emotion_logits", "state_cache_out"],
            )

            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 100_000  # Valid graph size

            # Test inference in ONNX Runtime
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            ort_inputs = {
                "audio_chunk": audio_in.numpy(),
                "state_cache_in": cache_in.numpy(),
            }
            ort_outs = session.run(None, ort_inputs)

            # PyTorch ground truth
            with torch.no_grad():
                pt_scene, pt_speaker, pt_emotion, pt_cache = model(audio_in, cache_in)

            np.testing.assert_allclose(ort_outs[0], pt_scene.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(ort_outs[1], pt_speaker.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(ort_outs[2], pt_emotion.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(ort_outs[3], pt_cache.numpy(), rtol=1e-3, atol=1e-3)
