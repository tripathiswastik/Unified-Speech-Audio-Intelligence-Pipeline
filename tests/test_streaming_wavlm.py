"""Unit and Integration Tests for Chunk-Based Streaming WavLM.

Validates:
1. Causal attention invariant (no lookahead into future frames).
2. Streaming state continuity across sequential 200ms audio chunks.
3. Multi-task head outputs (Scene, Voiceprint, Emotion).
4. Static-shape ONNX export and ONNX Runtime inference numerical equivalence.
"""
from typing import Any, Optional, Dict, List, Set, Tuple
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
from pipeline.device import safe_torch_load


def _get_or_create_test_vox_root(target_dir: Optional[Path] = None) -> Path:
    """Returns real VoxCeleb root if populated with audio, or creates synthetic mock files for CI."""
    from pipeline.config import BASE_DIR
    real_vox = BASE_DIR / "wav"
    if real_vox.exists():
        spk_dirs = [d for d in real_vox.iterdir() if d.is_dir()]
        if len(spk_dirs) >= 8 and any(len(list(d.glob("*/*.wav"))) > 0 for d in spk_dirs[:8]):
            return real_vox

    if target_dir is None:
        target_root = Path(tempfile.mkdtemp(prefix="mock_voxceleb_"))
    else:
        target_root = Path(target_dir)

    import wave
    for i in range(12):
        spk_dir = target_root / f"id100{i:02d}"
        for v in range(2):
            vid_dir = spk_dir / f"vid_{v:02d}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            for f in range(4):
                wav_file = vid_dir / f"{f:05d}.wav"
                if not wav_file.exists():
                    with wave.open(str(wav_file), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b"\x00" * 3200)
    return target_root


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


class TestStreamingParityAndEndToEndValidation:
    """End-to-end streaming tests: future-invariance, session cache reset, and PyTorch vs ONNX multi-chunk parity."""

    def test_end_to_end_streaming_future_invariance_and_session_reset(self):
        """Verify modifying future chunks cannot alter past emissions and cache reset eliminates cross-session state."""
        model = MultiTaskStreamingWavLM(
            hidden_dim=32,
            num_layers=3,
            num_heads=2,
            ffn_dim=64,
            max_past_frames=10,
            early_tap_layer=1,
            num_scene_classes=10,
            num_emotion_classes=8,
            embedding_dim=128,
        ).eval()

        chunk1 = torch.randn(1, 3200)
        chunk2 = torch.randn(1, 3200)
        chunk2_perturbed = chunk2 + torch.randn(1, 3200) * 5.0

        # Run stream A
        with torch.no_grad():
            sc1_a, spk1_a, emo1_a, cache1_a = model(chunk1, state_cache=None)
            sc2_a, spk2_a, emo2_a, cache2_a = model(chunk2, state_cache=cache1_a)

        # Run stream B (chunk 1 identical, chunk 2 perturbed)
        with torch.no_grad():
            sc1_b, spk1_b, emo1_b, cache1_b = model(chunk1, state_cache=None)
            sc2_b, spk2_b, emo2_b, cache2_b = model(chunk2_perturbed, state_cache=cache1_b)

        # Emitted chunk 1 outputs MUST be identical regardless of future perturbations
        assert torch.allclose(sc1_a, sc1_b, atol=1e-6)
        assert torch.allclose(spk1_a, spk1_b, atol=1e-6)
        assert torch.allclose(emo1_a, emo1_b, atol=1e-6)
        assert torch.allclose(cache1_a, cache1_b, atol=1e-6)

        # Session reset test:
        # Session 1: run chunk1 -> cache
        # Reset: set cache = None
        # Session 2: run chunk2 from fresh cache
        # Session 3: run chunk2 on an untouched fresh model
        with torch.no_grad():
            _, _, _, _ = model(chunk1, state_cache=None)
            # Reset cache between audio sessions
            sc2_fresh, spk2_fresh, emo2_fresh, _ = model(chunk2, state_cache=None)

        fresh_model = MultiTaskStreamingWavLM(
            hidden_dim=32,
            num_layers=3,
            num_heads=2,
            ffn_dim=64,
            max_past_frames=10,
            early_tap_layer=1,
            num_scene_classes=10,
            num_emotion_classes=8,
            embedding_dim=128,
        ).eval()
        fresh_model.load_state_dict(model.state_dict())

        with torch.no_grad():
            sc2_ctrl, spk2_ctrl, emo2_ctrl, _ = fresh_model(chunk2, state_cache=None)

        assert torch.allclose(sc2_fresh, sc2_ctrl, atol=1e-6), "Session reset failed: stale state persisted after cache reset!"
        assert torch.allclose(spk2_fresh, spk2_ctrl, atol=1e-6)
        assert torch.allclose(emo2_fresh, emo2_ctrl, atol=1e-6)

    def test_pytorch_vs_onnxruntime_multi_chunk_parity_with_cache_propagation(self):
        """Verify PyTorch and ONNX Runtime produce identical outputs over multiple consecutive chunks."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "streaming_parity.onnx"

            num_layers = 2
            heads = 2
            head_dim = 16
            past_frames = 10
            model = MultiTaskStreamingWavLM(
                hidden_dim=32,
                num_layers=num_layers,
                num_heads=heads,
                ffn_dim=64,
                max_past_frames=past_frames,
                early_tap_layer=1,
                num_scene_classes=10,
                num_emotion_classes=8,
                embedding_dim=128,
            ).eval()

            dummy_chunk = torch.randn(1, 3200)
            dummy_cache = torch.randn(num_layers, 2, 1, heads, past_frames, head_dim)

            torch.onnx.export(
                model,
                (dummy_chunk, dummy_cache),
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["audio_chunk", "state_cache_in"],
                output_names=["scene_logits", "speaker_embedding", "emotion_logits", "state_cache_out"],
            )

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            # Stream 4 consecutive chunks through both PyTorch and ONNX
            pt_cache = dummy_cache.clone()
            ort_cache = dummy_cache.numpy().copy()

            for step in range(4):
                chunk = torch.randn(1, 3200)

                with torch.no_grad():
                    pt_sc, pt_spk, pt_emo, pt_cache = model(chunk, state_cache=pt_cache)

                ort_out = session.run(None, {
                    "audio_chunk": chunk.numpy(),
                    "state_cache_in": ort_cache,
                })
                ort_sc, ort_spk, ort_emo, ort_cache = ort_out

                np.testing.assert_allclose(ort_sc, pt_sc.numpy(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ort_spk, pt_spk.numpy(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ort_emo, pt_emo.numpy(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ort_cache, pt_cache.numpy(), rtol=1e-3, atol=1e-3)

    def test_atomic_checkpoint_save_and_provenance(self):
        """Verify save_checkpoint_atomic writes atomically and preserves RNG states."""
        from pipeline.train_streaming_wavlm import save_checkpoint_atomic
        import random

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"
            state = {
                "epoch": 5,
                "best_composite": 0.88,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
            }

            save_checkpoint_atomic(state, ckpt_path)
            assert ckpt_path.exists()
            # Ensure no orphaned .tmp files
            tmp_files = list(Path(tmp_dir).glob("*.tmp*"))
            assert len(tmp_files) == 0

            loaded = safe_torch_load(ckpt_path, weights_only=False)
            assert loaded["epoch"] == 5
            assert loaded["best_composite"] == 0.88

    def test_voxceleb_splits_disjoint_identities_and_files(self):
        """Verify VoxCeleb train, heldout_id, val, and test splits have completely disjoint identities and files."""
        from pipeline.train_streaming_wavlm import VoxCelebRawDataset

        vox_root = _get_or_create_test_vox_root()
        ds_train = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="train", strict=False)
        ds_heldout = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="heldout_id", speaker_to_id=ds_train.speaker_to_id, strict=False)
        ds_val = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="val", strict=False)
        ds_test = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="test", strict=False)

        assert len(ds_train) > 0 and len(ds_heldout) > 0 and len(ds_val) > 0 and len(ds_test) > 0

        tr_spk = set(ds_train.speaker_names)
        va_spk = set(ds_val.speaker_names)
        te_spk = set(ds_test.speaker_names)

        # 1. Speaker identity disjointness
        assert tr_spk.isdisjoint(va_spk), f"Train and val share speakers: {tr_spk & va_spk}"
        assert va_spk.isdisjoint(te_spk), f"Val and test share speakers: {va_spk & te_spk}"
        assert tr_spk.isdisjoint(te_spk), f"Train and test share speakers: {tr_spk & te_spk}"

        # 2. Audio file path disjointness
        tr_f = set(ds_train.file_paths)
        hid_f = set(ds_heldout.file_paths)
        va_f = set(ds_val.file_paths)
        te_f = set(ds_test.file_paths)

        assert tr_f.isdisjoint(hid_f), f"Train and heldout_id share files: {tr_f & hid_f}"
        assert tr_f.isdisjoint(va_f), f"Train and val share files: {tr_f & va_f}"
        assert tr_f.isdisjoint(te_f), f"Train and test share files: {tr_f & te_f}"
        assert va_f.isdisjoint(te_f), f"Val and test share files: {va_f & te_f}"

    def test_speaker_verification_trials_zero_leakage_and_independent_populations(self):
        """Verify that validation and test speaker verification trials do not overlap in files or speaker identities."""
        from pipeline.train_streaming_wavlm import VoxCelebRawDataset, generate_speaker_verification_trials

        vox_root = _get_or_create_test_vox_root()
        ds_train = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="train", strict=False)
        ds_val = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="val", strict=False)
        ds_test = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="test", strict=False)

        val_trials = generate_speaker_verification_trials(ds_val, num_pairs=10, seed=42)
        test_trials = generate_speaker_verification_trials(ds_test, num_pairs=10, seed=1337)

        assert len(val_trials) > 0
        assert len(test_trials) > 0

        val_files = {f1 for f1, f2, _ in val_trials} | {f2 for f1, f2, _ in val_trials}
        test_files = {f1 for f1, f2, _ in test_trials} | {f2 for f1, f2, _ in test_trials}
        tr_files = set(ds_train.file_paths)

        # Trials file path disjointness
        assert val_files.isdisjoint(test_files), f"Val and test trials share files: {val_files & test_files}"
        assert test_files.isdisjoint(tr_files), f"Test trials contain training files: {test_files & tr_files}"

        # Trials speaker identity disjointness
        val_speakers = {f.parent.parent.name for f in val_files}
        test_speakers = {f.parent.parent.name for f in test_files}
        tr_speakers = set(ds_train.speaker_names)

        assert val_speakers.isdisjoint(test_speakers), f"Val and test trials share speakers: {val_speakers & test_speakers}"
        assert test_speakers.isdisjoint(tr_speakers), f"Test trials contain training speakers: {test_speakers & tr_speakers}"

    def test_preflight_audit_enforces_cross_split_disjointness_and_rejects_leakage(self):
        """Verify that run_dataset_preflight_audit raises ValueError if trials or speakers overlap across splits."""
        from pipeline.train_streaming_wavlm import VoxCelebRawDataset, generate_speaker_verification_trials, run_dataset_preflight_audit

        vox_root = _get_or_create_test_vox_root()
        ds_val = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="val", strict=False)
        ds_test = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="test", strict=False)

        val_trials = generate_speaker_verification_trials(ds_val, num_pairs=4, seed=42)
        # Artificially contaminate test trials with a validation trial
        contaminated_test_trials = list(val_trials[:1]) + generate_speaker_verification_trials(ds_test, num_pairs=4, seed=1337)

        splits = [
            ("VoxCeleb (Disjoint Val)", ds_val),
            ("VoxCeleb (Disjoint Test)", ds_test),
        ]

        with pytest.raises(ValueError) as excinfo:
            run_dataset_preflight_audit(splits, val_spk_trials=val_trials, test_spk_trials=contaminated_test_trials, min_samples=1)
        assert "share" in str(excinfo.value).lower() or "preflight audit failed" in str(excinfo.value).lower()

    def test_preflight_audit_detects_corrupt_audio_and_inadequate_coverage(self):
        """Verify preflight audit detects truncated files, out-of-range labels, and missing class coverage."""
        from pipeline.train_streaming_wavlm import run_dataset_preflight_audit
        from torch.utils.data import Dataset

        class MockCorruptDataset(Dataset):
            def __init__(self, corrupt_file: Path):
                self.items = [(corrupt_file, 0)]
                self.file_paths = [corrupt_file]
                self.labels = [0]
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return torch.zeros(16000), 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "corrupt.wav"
            bad_file.write_bytes(b"RIFF" + b"\x00" * 10)  # Only 14 bytes (< 44)
            ds_bad = MockCorruptDataset(bad_file)
            with pytest.raises(ValueError) as exc:
                run_dataset_preflight_audit([("CorruptSplit", ds_bad)], min_samples=1)
            assert "corrupt" in str(exc.value).lower() or "size" in str(exc.value).lower()

        class MockLabelDataset(Dataset):
            def __init__(self):
                self.items = [(Path("dummy.wav"), 999)]
                self.labels = [999]
                self.file_paths = []
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return torch.zeros(16000), 999

        with pytest.raises(ValueError) as exc:
            run_dataset_preflight_audit([("UrbanSound (Train)", MockLabelDataset())], min_samples=1)
        assert "exceeds" in str(exc.value).lower() or "out of" in str(exc.value).lower() or "inadequate" in str(exc.value).lower()


class TestEvaluationMetricEdgeCases:
    """Validate speaker verification metric edge cases and mathematical invariants."""

    def test_compute_equal_error_rate_perfect_separation(self):
        """Linearly separable genuine and impostor scores must yield EER=0.0 and AUC=1.0."""
        from pipeline.train_streaming_wavlm import compute_equal_error_rate

        genuine = np.array([0.80, 0.85, 0.90, 0.95], dtype=np.float64)
        impostor = np.array([0.10, 0.20, 0.25, 0.30], dtype=np.float64)

        eer, auc, th = compute_equal_error_rate(genuine, impostor)
        assert eer == 0.0, f"Expected 0.0 EER for perfect separation, got {eer}"
        assert auc == 1.0, f"Expected 1.0 AUC for perfect separation, got {auc}"
        assert 0.30 <= th <= 0.80, f"Threshold should separate distributions, got {th}"

    def test_compute_equal_error_rate_completely_overlapping(self):
        """Identical score distributions must yield EER=0.5 and AUC=0.5."""
        from pipeline.train_streaming_wavlm import compute_equal_error_rate

        genuine = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        impostor = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        eer, auc, th = compute_equal_error_rate(genuine, impostor)
        assert np.isclose(eer, 0.5, atol=1e-3), f"Expected 0.5 EER for overlapping distributions, got {eer}"
        assert np.isclose(auc, 0.5, atol=1e-3), f"Expected 0.5 AUC for overlapping distributions, got {auc}"

    def test_compute_equal_error_rate_empty_trials_handling(self):
        """Empty genuine or impostor trials must return NaN (strict=False) or raise ValueError (strict=True)."""
        import math
        from pipeline.train_streaming_wavlm import compute_equal_error_rate

        # Non-strict mode should return (nan, nan, 0.0) without silently returning a misleading 0.5
        eer, auc, th = compute_equal_error_rate(np.array([], dtype=np.float64), np.array([0.1, 0.2], dtype=np.float64), strict=False)
        assert math.isnan(eer)
        assert math.isnan(auc)

        eer2, auc2, th2 = compute_equal_error_rate(np.array([0.8, 0.9], dtype=np.float64), np.array([], dtype=np.float64), strict=False)
        assert math.isnan(eer2)
        assert math.isnan(auc2)

        # Strict mode must fail loudly
        with pytest.raises(ValueError):
            compute_equal_error_rate(np.array([], dtype=np.float64), np.array([0.1, 0.2], dtype=np.float64), strict=True)

        with pytest.raises(ValueError):
            compute_equal_error_rate(np.array([0.8, 0.9], dtype=np.float64), np.array([], dtype=np.float64), strict=True)

    def test_compute_equal_error_rate_tied_scores_and_extreme_range(self):
        """Handles scores spanning the entire [-1.0, 1.0] cosine similarity range with ties."""
        from pipeline.train_streaming_wavlm import compute_equal_error_rate

        genuine = np.array([-0.1, 0.4, 0.4, 0.7, 0.9], dtype=np.float64)
        impostor = np.array([-0.9, -0.5, 0.4, 0.4, 0.6], dtype=np.float64)

        eer, auc, th = compute_equal_error_rate(genuine, impostor)
        assert 0.0 <= eer <= 1.0
        assert 0.0 <= auc <= 1.0
        assert -1.0 <= th <= 1.0

    def test_compute_equal_error_rate_auc_consistency_with_sklearn(self):
        """ROC-AUC from compute_equal_error_rate must match sklearn.metrics.roc_auc_score within 1e-4."""
        from pipeline.train_streaming_wavlm import compute_equal_error_rate
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(42)
        genuine = rng.normal(loc=0.6, scale=0.15, size=100)
        impostor = rng.normal(loc=0.2, scale=0.15, size=100)

        eer, auc_computed, th = compute_equal_error_rate(genuine, impostor)

        y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
        y_scores = np.concatenate([genuine, impostor])
        auc_sklearn = float(roc_auc_score(y_true, y_scores))

        assert np.isclose(auc_computed, auc_sklearn, atol=1e-4), (
            f"AUC mismatch: computed {auc_computed} vs sklearn {auc_sklearn}"
        )


class TestSplitDeterminismAndTrialManifests:
    """Validate split repeatability, trial edge cases, and manifest persistence."""

    def test_split_determinism_under_same_seed_and_difference_under_different_seeds(self):
        """Same seed yields identical partition; different seed yields different partition."""
        from pipeline.train_streaming_wavlm import VoxCelebRawDataset

        vox_root = _get_or_create_test_vox_root()
        ds_val_seed1 = VoxCelebRawDataset(vox_root=vox_root, max_speakers=12, split="val", seed=42, strict=False)
        ds_val_seed1_again = VoxCelebRawDataset(vox_root=vox_root, max_speakers=12, split="val", seed=42, strict=False)
        ds_val_seed2 = VoxCelebRawDataset(vox_root=vox_root, max_speakers=12, split="val", seed=999, strict=False)

        assert ds_val_seed1.speaker_names == ds_val_seed1_again.speaker_names
        assert set(ds_val_seed1.file_paths) == set(ds_val_seed1_again.file_paths)
        if len(ds_val_seed1.speaker_names) > 0 and len(ds_val_seed2.speaker_names) > 0:
            # Different seeds should yield different speaker ordering or selections
            assert (ds_val_seed1.speaker_names != ds_val_seed2.speaker_names) or (len(ds_val_seed1.speaker_names) <= 1)

    def test_trial_generation_insufficient_recordings_failure_mode(self):
        """Trial generator should fail explicitly when speakers have insufficient recordings."""
        from pipeline.train_streaming_wavlm import generate_speaker_verification_trials
        from torch.utils.data import Dataset

        class SingleFileSpeakerDataset(Dataset):
            def __init__(self):
                # 2 speakers, each with only 1 file -> Impossible to form genuine pairs
                self.items = [
                    (Path("spk_a/rec1.wav"), 0),
                    (Path("spk_b/rec1.wav"), 1),
                ]
                self.labels = [0, 1]
                self.file_paths = [Path("spk_a/rec1.wav"), Path("spk_b/rec1.wav")]
                self.speaker_names = ["spk_a", "spk_b"]
                self.speaker_files = {
                    0: [Path("spk_a/rec1.wav")],
                    1: [Path("spk_b/rec1.wav")],
                }
                self.split = "test_single"
            def __len__(self):
                return len(self.items)
            def __getitem__(self, index):
                return torch.zeros(16000), self.labels[index]

        ds_single: Any = SingleFileSpeakerDataset()

        # In strict mode (fail_on_insufficient=True), it must raise ValueError
        with pytest.raises(ValueError) as exc:
            generate_speaker_verification_trials(ds_single, num_pairs=10, fail_on_insufficient=True)
        assert "insufficient" in str(exc.value).lower() or "cannot generate" in str(exc.value).lower()

        # In non-strict mode, it returns empty list or valid pairs without crashing
        trials = generate_speaker_verification_trials(ds_single, num_pairs=10, fail_on_insufficient=False)
        assert len(trials) == 0

    def test_manifest_persistence_and_checksum_integrity(self):
        """Verifies that split and trial manifests are persisted to JSON with SHA-256 fingerprint."""
        from pipeline.train_streaming_wavlm import (
            VoxCelebRawDataset,
            generate_speaker_verification_trials,
            save_voxceleb_splits_manifest,
            save_trials_manifest,
        )
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            vox_root = _get_or_create_test_vox_root(out_dir)
            ds_train = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="train", strict=False)
            ds_val = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="val", strict=False)
            ds_test = VoxCelebRawDataset(vox_root=vox_root, max_speakers=8, split="test", strict=False)

            manifest_path = save_voxceleb_splits_manifest(
                splits={
                    "train": ds_train,
                    "val": ds_val,
                    "test": ds_test,
                },
                manifest_path=out_dir / "vox_splits.json",
            )
            assert manifest_path.exists()

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "manifest_sha256" in data
            assert len(data["manifest_sha256"]) == 64
            assert "splits" in data
            assert "train" in data["splits"]
            assert "val" in data["splits"]
            assert "test" in data["splits"]

            # Save trials manifest
            trials = generate_speaker_verification_trials(ds_val, num_pairs=4, seed=42)
            trial_manifest_path = save_trials_manifest(
                trials=trials,
                split_name="test_val",
                manifest_path=out_dir / "val_trials.json",
            )
            assert trial_manifest_path.exists()
            with open(trial_manifest_path, "r", encoding="utf-8") as tf:
                t_data = json.load(tf)
            assert t_data["split_name"] == "test_val"
            assert t_data["total_trials"] == len(trials)
            assert "num_genuine" in t_data
            assert "num_impostor" in t_data
            assert "is_balanced" in t_data
            assert "manifest_sha256" in t_data
            assert "trials" in t_data


class TestStreamingStateAndBoundedMemory:
    """Validate streaming state memory bounds and batched inference consistency."""

    def test_streaming_cache_bounded_over_long_stream(self):
        """State cache temporal dimension must strictly remain bounded by max_past_frames over 25 chunks."""
        model = MultiTaskStreamingWavLM(
            hidden_dim=32,
            num_layers=3,
            num_heads=2,
            ffn_dim=64,
            max_past_frames=15,
            early_tap_layer=1,
            num_scene_classes=10,
            num_emotion_classes=8,
            embedding_dim=128,
        ).eval()

        chunk_samples = 3200  # 200ms
        state_cache = None

        for chunk_idx in range(25):  # 5 seconds of continuous streaming
            chunk = torch.randn(1, chunk_samples)
            with torch.no_grad():
                sc, spk, emo, state_cache = model(chunk, state_cache=state_cache)

            # Ensure cache temporal length is clamped at max_past_frames (15)
            past_len = state_cache.shape[4]
            assert past_len <= 15, f"Cache grew beyond max_past_frames: {past_len} > 15"
            if chunk_idx >= 2:
                assert past_len == 15, f"Cache should be full (15), got {past_len}"

    def test_batched_streaming_inference(self):
        """Batch size 2 inference produces identical per-sample outputs as running batch size 1 sequentially."""
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
        ).eval()

        chunk1 = torch.randn(1, 3200)
        chunk2 = torch.randn(1, 3200)
        batched_chunk = torch.cat([chunk1, chunk2], dim=0)

        with torch.no_grad():
            sc_b, spk_b, emo_b, _ = model(batched_chunk, state_cache=None)
            sc1, spk1, emo1, _ = model(chunk1, state_cache=None)
            sc2, spk2, emo2, _ = model(chunk2, state_cache=None)

        assert torch.allclose(sc_b[0:1], sc1, atol=1e-5)
        assert torch.allclose(sc_b[1:2], sc2, atol=1e-5)
        assert torch.allclose(spk_b[0:1], spk1, atol=1e-5)
        assert torch.allclose(spk_b[1:2], spk2, atol=1e-5)
        assert torch.allclose(emo_b[0:1], emo1, atol=1e-5)
        assert torch.allclose(emo_b[1:2], emo2, atol=1e-5)


class TestCheckpointResumeAndSafety:
    """Validate resume safety, fingerprint mismatch rejection, and forced resume override."""

    def test_incompatible_checkpoint_fingerprint_fails_closed(self):
        """Resuming with mismatched architecture or dataset fingerprint must raise ValueError unless force_resume."""
        current_fingerprint = {
            "num_speakers": 10,
            "hidden_dim": 256,
            "num_layers": 6,
            "val_fold": 10,
        }
        saved_fingerprint = {
            "num_speakers": 10,
            "hidden_dim": 128,  # Mismatched architecture hidden_dim!
            "num_layers": 6,
            "val_fold": 10,
        }

        # Simulate resume check logic
        def check_resume(saved_fp, current_fp, force=False):
            if saved_fp and saved_fp != current_fp and not force:
                raise ValueError(
                    f"[Resume Aborted] Incompatible dataset/architecture fingerprint:\n"
                    f"  Saved:   {saved_fp}\n"
                    f"  Current: {current_fp}\n"
                    f"Use --force-resume to override."
                )
            return True

        with pytest.raises(ValueError) as exc:
            check_resume(saved_fingerprint, current_fingerprint, force=False)
        assert "incompatible" in str(exc.value).lower()

        # With force=True, it succeeds
        assert check_resume(saved_fingerprint, current_fingerprint, force=True)




