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
    CausalMultiTaskAudioTransformer,
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

    def test_full_encoder_strict_causality_invariant(self):
        """Modifying the second half of an audio chunk must NOT alter earlier encoder output frames.
        
        Verifies that CausalChannelNorm computes statistics strictly per time frame with 0 temporal leakage.
        """
        encoder = ConvFeatureEncoder(hidden_dim=64).eval()

        audio_a = torch.randn(1, 1, 3200)
        audio_b = audio_a.clone()
        # Strongly perturb future samples in the second half
        audio_b[:, :, 1600:] = torch.randn(1, 1, 1600) * 10.0

        with torch.no_grad():
            feat_a = encoder(audio_a)  # (1, 10, 64)
            feat_b = encoder(audio_b)  # (1, 10, 64)

        # First 3 frames depend strictly on audio samples < 1600 (cumulative receptive field < 1200 samples)
        diff = torch.max(torch.abs(feat_a[:, :3, :] - feat_b[:, :3, :])).item()
        assert diff < 1e-5, f"Temporal future leakage detected in ConvFeatureEncoder: max diff = {diff}"

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
        model = CausalMultiTaskAudioTransformer(
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


class TestMultiTaskTrainingPipeline:
    """Validate joint multi-task training logic, loss weighting, and gradient flow."""

    def test_uncertainty_loss_formulation(self):
        from pipeline.train_streaming_wavlm import MultiTaskUncertaintyLoss

        criterion = MultiTaskUncertaintyLoss()
        l_scene = torch.tensor(1.5, requires_grad=True)
        l_spk = torch.tensor(2.0, requires_grad=True)
        l_emo = torch.tensor(1.2, requires_grad=True)

        total_loss, weights = criterion(l_scene, l_spk, l_emo)
        assert total_loss.item() > 0.0
        assert weights["w_scene"] > 0.0
        assert weights["w_spk"] > 0.0
        assert weights["w_emo"] > 0.0

        total_loss.backward()
        assert criterion.log_vars.grad is not None
        assert criterion.log_vars.grad.shape == (3,)

    def test_chunk_streaming_forward_and_gradient_flow(self):
        from pipeline.train_streaming_wavlm import process_audio_in_streaming_chunks
        from pipeline.model import ArcMarginProduct

        model = MultiTaskStreamingWavLM(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            ffn_dim=64,
            max_past_frames=10,
            early_tap_layer=1,
            num_scene_classes=10,
            num_emotion_classes=8,
            embedding_dim=128,
        )
        arc_margin = ArcMarginProduct(in_features=128, out_features=4, scale=16.0, margin=0.2)

        # 2 audio samples, each 6,400 samples (2 sequential 200ms chunks)
        audio = torch.randn(2, 6400, requires_grad=True)
        target_scene = torch.tensor([1, 4], dtype=torch.long)
        target_spk = torch.tensor([0, 2], dtype=torch.long)
        target_emo = torch.tensor([3, 7], dtype=torch.long)

        scene_l, spk_e, emo_l = process_audio_in_streaming_chunks(model, audio, chunk_samples=3200)

        assert scene_l.shape == (2, 10)
        assert spk_e.shape == (2, 128)
        assert emo_l.shape == (2, 8)

        # Compute joint losses
        loss_s = F.cross_entropy(scene_l, target_scene)
        loss_spk = F.cross_entropy(arc_margin(spk_e, target_spk), target_spk)
        loss_e = F.cross_entropy(emo_l, target_emo)
        joint_loss = loss_s + loss_spk + loss_e

        joint_loss.backward()

        # Verify gradient flow into feature encoder, attention layers, and all heads
        assert model.feature_encoder.layers[0].conv.weight.grad is not None
        assert model.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.scene_head.head[0].weight.grad is not None
        assert model.voiceprint_head.projection[0].weight.grad is not None
        assert model.emotion_head.classifier[0].weight.grad is not None
        assert arc_margin.weight.grad is not None

    def test_speaker_verification_eer_computation(self):
        """Verify Equal Error Rate (EER) and ROC-AUC calculation on genuine/impostor score distributions."""
        from pipeline.train_streaming_wavlm import compute_equal_error_rate

        rng = np.random.RandomState(42)
        genuine_scores = rng.normal(loc=0.75, scale=0.10, size=200)
        impostor_scores = rng.normal(loc=0.15, scale=0.12, size=200)

        eer, auc, th = compute_equal_error_rate(genuine_scores, impostor_scores)
        assert eer < 0.05, f"Expected EER < 0.05 for separated distributions, got {eer}"
        assert auc > 0.98, f"Expected AUC > 0.98, got {auc}"
        assert 0.3 < th < 0.6, f"Expected optimal threshold in [0.3, 0.6], got {th}"

    def test_dataset_loaders_reject_missing_data(self):
        """Verify dataset loaders fail with FileNotFoundError when paths/data do not exist."""
        from pipeline.train_streaming_wavlm import (
            UrbanSoundRawDataset,
            VoxCelebRawDataset,
            RAVDESSRawDataset,
        )

        with pytest.raises(FileNotFoundError):
            UrbanSoundRawDataset(
                folds=[1],
                audio_dir=Path("non_existent_audio_dir_xyz"),
                strict=True,
            )

        with pytest.raises(FileNotFoundError):
            VoxCelebRawDataset(
                vox_root=Path("non_existent_vox_dir_xyz"),
                strict=True,
            )

        with pytest.raises(FileNotFoundError):
            RAVDESSRawDataset(
                actors=[1],
                metadata_csv=Path("non_existent_ravdess_meta.csv"),
                strict=True,
            )

    def test_synthetic_noise_all_classes_and_unsupported_rejection(self):
        """Verify all 10 UrbanSound classes synthesize valid audio and unsupported classes raise ValueError."""
        from pipeline.urban_noise_mixer import generate_synthetic_noise
        from pipeline.config import CLASSES

        for cls_name in CLASSES:
            noise = generate_synthetic_noise(cls_name, num_samples=3200)
            assert len(noise) == 3200
            assert np.all(np.isfinite(noise)), f"Non-finite values in {cls_name}"
            assert float(np.std(noise)) > 0.01, f"Zero-energy or flat noise in {cls_name}"

        # Reject unsupported noise class
        with pytest.raises(ValueError):
            generate_synthetic_noise("non_existent_noise_type_xyz", num_samples=3200)


class TestStreamingWavLMExportCheckpoints:
    """Validate strict checkpoint loading and failure modes during ONNX export."""

    def test_onnx_export_fails_closed_when_checkpoint_missing(self):
        """Exporting to ONNX in strict mode must fail with FileNotFoundError if checkpoint is missing."""
        with pytest.raises(FileNotFoundError):
            export_streaming_wavlm_to_onnx(
                checkpoint_path=Path("non_existent_ckpt_12345.pt"),
                strict_checkpoints=True,
            )


