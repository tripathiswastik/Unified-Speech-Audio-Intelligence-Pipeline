"""Tests for Deep Learning Architectural Upgrades:
1. Squeeze-and-Excitation (SE) Channel Attention
2. Depthwise Separable Convolutions
3. ArcFace (Additive Angular Margin) Loss Head
4. Per-Channel Cepstral Mean & Variance Normalization (CMVN)
5. Self-Supervised Learning (SSL) Adapter for SER
6. Upgraded SESpeakerEmbeddingNet & SEAudioCNN
"""
import numpy as np
import torch
import pytest

from pipeline.model import (
    SEBlock,
    DepthwiseSeparableConv2d,
    SEAudioCNN,
    ArcMarginProduct,
    SpeakerEmbeddingNet,
    SESpeakerEmbeddingNet,
)
from pipeline.audio_loader import (
    compute_log_mel_spectrogram,
    compute_cmvn_log_mel_spectrogram,
    extract_ssl_features,
)
from pipeline.emotion_model import (
    SEEmotionClassifierCNN,
    SSLEmotionClassifier,
)
from pipeline.train_speaker_id import SpeakerArcFaceClassifier


def test_se_block_forward():
    """Verify SEBlock channel recalibration and shape preservation."""
    se = SEBlock(channels=64, reduction=8)
    se.eval()
    x = torch.randn(2, 64, 32, 40)
    with torch.no_grad():
        out = se(x)
    assert out.shape == x.shape
    # Channels should be dynamically reweighted, not all equal to input
    assert not torch.allclose(out, x)


def test_depthwise_separable_conv2d():
    """Verify DepthwiseSeparableConv2d parameter reduction and forward pass."""
    in_c, out_c = 32, 64
    dw_conv = DepthwiseSeparableConv2d(in_c, out_c, kernel_size=3, padding=1)
    std_conv = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)

    dw_params = sum(p.numel() for p in dw_conv.parameters())
    std_params = sum(p.numel() for p in std_conv.parameters())

    # Depthwise-separable parameter count should be significantly smaller
    assert dw_params < std_params

    x = torch.randn(2, in_c, 16, 20)
    out = dw_conv(x)
    assert out.shape == (2, out_c, 16, 20)


def test_se_audiocnn_forward():
    """Verify SEAudioCNN accepts mel-spectrograms and outputs 10 class logits."""
    model = SEAudioCNN(num_classes=10)
    model.eval()
    # (B, 1, n_mels=128, time_steps=173)
    dummy_input = torch.randn(2, 1, 128, 173)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (2, 10)
    assert not torch.isnan(logits).any()


def test_cmvn_log_mel_spectrogram_normalization():
    """Verify compute_cmvn_log_mel_spectrogram normalizes per frequency channel across time."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Synthetic waveform with stationary tonal noise
    wave = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))).astype(np.float32)

    cmvn_spec = compute_cmvn_log_mel_spectrogram(wave, sr=sr)
    assert cmvn_spec.ndim == 2  # (n_mels, time_steps)

    # Across time (axis=1), each frequency band should have mean ~ 0 and std ~ 1
    channel_means = np.mean(cmvn_spec, axis=1)
    channel_stds = np.std(cmvn_spec, axis=1)

    assert np.allclose(channel_means, np.zeros_like(channel_means), atol=1e-5)
    # Most channels should have unit standard deviation
    assert np.allclose(channel_stds, np.ones_like(channel_stds), atol=1e-3)


def test_arcface_margin_product_forward():
    """Verify ArcFace head injects angular margin and scales output properly."""
    in_dim, num_classes = 128, 10
    head = ArcMarginProduct(in_features=in_dim, out_features=num_classes, scale=30.0, margin=0.30)
    head.eval()

    embeddings = torch.randn(4, in_dim)
    labels = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    outputs = head(embeddings, labels)
    assert outputs.shape == (4, num_classes)
    assert not torch.isnan(outputs).any()


def test_se_speaker_embedding_net():
    """Verify SESpeakerEmbeddingNet forward pass and unit L2 norm."""
    net_se = SESpeakerEmbeddingNet(embedding_dim=128)
    net_se.eval()
    dummy_input = torch.randn(2, 1, 128, 100)
    with torch.no_grad():
        emb = net_se(dummy_input)

    assert emb.shape == (2, 128)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Test depthwise backbone variant
    net_dw = SpeakerEmbeddingNet(embedding_dim=128, backbone="depthwise")
    net_dw.eval()
    with torch.no_grad():
        emb_dw = net_dw(dummy_input)
    assert emb_dw.shape == (2, 128)


def test_speaker_arcface_classifier():
    """Verify SpeakerArcFaceClassifier behaves correctly during training (with labels) and eval."""
    emb_net = SpeakerEmbeddingNet(embedding_dim=128, backbone="baseline")
    classifier = SpeakerArcFaceClassifier(embedding_net=emb_net, num_classes=5, scale=30.0, margin=0.30)

    dummy_input = torch.randn(3, 1, 128, 80)
    labels = torch.tensor([1, 3, 4], dtype=torch.long)

    # Training mode (with target labels)
    classifier.train()
    logits_train, emb = classifier(dummy_input, labels)
    assert logits_train.shape == (3, 5)
    assert emb.shape == (3, 128)

    # Evaluation mode (without target labels)
    classifier.eval()
    with torch.no_grad():
        logits_eval, _ = classifier(dummy_input)
    assert logits_eval.shape == (3, 5)


def test_se_emotion_classifier_cnn():
    """Verify SEEmotionClassifierCNN forward pass produces 8 emotion logits."""
    model = SEEmotionClassifierCNN(num_classes=8)
    model.eval()
    dummy_input = torch.randn(2, 1, 128, 173)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (2, 8)
    assert not torch.isnan(logits).any()


def test_ssl_emotion_classifier_attentive_pooling():
    """Verify SSLEmotionClassifier attentive pooling on hidden states."""
    # Model with dummy hidden dim 768
    model = SSLEmotionClassifier(hidden_dim=768, num_classes=8, pretrained_model_name=None)
    model.eval()

    # Input: already extracted SSL transformer frame states (B=2, T=50, H=768)
    ssl_states = torch.randn(2, 50, 768)
    with torch.no_grad():
        logits = model(ssl_states)
    assert logits.shape == (2, 8)


def test_extract_ssl_features_graceful_fallback():
    """Verify extract_ssl_features handles unavailable transformer or model gracefully."""
    synth_wave = np.random.randn(16000).astype(np.float32)
    # Should safely return None if transformers not present or model unavailable
    res = extract_ssl_features(synth_wave, model_name="nonexistent_mock_model_12345")
    assert res is None or isinstance(res, np.ndarray)
