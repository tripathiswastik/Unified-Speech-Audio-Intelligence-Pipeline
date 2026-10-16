"""Chunk-Based Streaming WavLM Multi-Task Backbone.

Architectural Highlights:
1. Causal / Masked Attention: Restricts attention to current and past frames, eliminating future lookahead.
2. Chunk-State KV-Cache: Accepts small audio chunks (e.g. 200ms = 3,200 samples) along with a hidden-state
   cache carried forward from the preceding chunk.
3. Multi-Task Shared Backbone:
   - Early Layer Taps (Layer 2): Extracts acoustic textures and noise for Environmental Scene Classification (UrbanSound/ESC-50).
   - Deep Layer Taps (Layer 6): Extracts prosody and identity for Voiceprint Verification (128-d L2 vector) and Speech Emotion Recognition.
4. Edge Runtime Ready (ONNX / TensorRT): Strict static tensor shapes, standard GEMM / LayerNorm / Conv1D ops.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import CHECKPOINTS_DIR, CLASSES, SAMPLE_RATE
from pipeline.emotion_dataset import EMOTIONS


class CausalConv1d(nn.Module):
    """1D Convolution with causal left-padding to prevent future lookahead."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad only (causal padding)
        pad = self.kernel_size - self.stride
        if pad > 0:
            x = F.pad(x, (pad, 0))
        return self.conv(x)


class ConvFeatureEncoder(nn.Module):
    """Temporal convolutional feature extractor downsampling raw 16kHz audio to frame embeddings.
    
    Transforms (B, 1, T_samples) -> (B, T_frames, hidden_dim).
    For a 200ms chunk (3,200 samples), outputs exactly 10 temporal frames (50 Hz frame rate).
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Stride breakdown: 5 x 4 x 4 x 4 = 320 total downsampling factor (16,000 / 320 = 50 frames/sec)
        self.layers = nn.Sequential(
            CausalConv1d(1, 64, kernel_size=10, stride=5),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            CausalConv1d(64, 128, kernel_size=8, stride=4),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            CausalConv1d(128, 192, kernel_size=8, stride=4),
            nn.GroupNorm(1, 192),
            nn.GELU(),
            CausalConv1d(192, hidden_dim, kernel_size=8, stride=4),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, 1, T_samples)
        feat = self.layers(waveform)  # (B, hidden_dim, T_frames)
        return feat.transpose(1, 2)  # (B, T_frames, hidden_dim)


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention with Causal Masking and fixed-size KV-Cache for streaming chunk ingestion."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, max_past_frames: int = 40):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.max_past_frames = max_past_frames

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with causal chunk attention and static KV-cache state update.
        
        Args:
            x: (B, T_chunk, hidden_dim)
            past_k: (B, num_heads, past_len, head_dim) or None
            past_v: (B, num_heads, past_len, head_dim) or None
            
        Returns:
            Tuple of (output, new_k_cache, new_v_cache)
        """
        B, T_curr, _ = x.shape

        q = self.q_proj(x).view(B, T_curr, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_curr, D)
        k = self.k_proj(x).view(B, T_curr, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_curr, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate with past cache along temporal dimension
        if past_k is not None and past_v is not None:
            full_k = torch.cat([past_k, k], dim=2)  # (B, H, T_total, D)
            full_v = torch.cat([past_v, v], dim=2)
        else:
            full_k = k
            full_v = v

        T_total = full_k.shape[2]
        T_past = T_total - T_curr

        # Attention scores: Q * K^T
        scores = torch.matmul(q, full_k.transpose(-1, -2)) * self.scale  # (B, H, T_curr, T_total)

        # Causal mask: Current queries can attend to all past frames + earlier frames in current chunk,
        # but CANNOT attend to future frames in current chunk.
        # Vectorized for ONNX / TensorRT graph tracing without dynamic slicing loops.
        row_idx = torch.arange(T_curr, device=x.device, dtype=torch.long).unsqueeze(1)
        col_idx = torch.arange(T_total, device=x.device, dtype=torch.long).unsqueeze(0)
        causal_mask = col_idx <= (T_past + row_idx)

        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-1e9"))
        attn_weights = F.softmax(scores, dim=-1)

        # Attention output: Attn * V
        attn_out = torch.matmul(attn_weights, full_v)  # (B, H, T_curr, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_curr, self.hidden_dim)
        output = self.out_proj(attn_out)

        # Update and boundedly truncate KV-cache (retain up to max_past_frames for static shape stability)
        new_k = full_k[:, :, -self.max_past_frames :, :].contiguous()
        new_v = full_v[:, :, -self.max_past_frames :, :].contiguous()

        return output, new_k, new_v


class StreamingWavLMBlock(nn.Module):
    """Transformer Encoder layer with causal self-attention, feed-forward network, and residual connections."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, ffn_dim: int = 512, max_past_frames: int = 40):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads=num_heads, max_past_frames=max_past_frames)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_x = self.ln1(x)
        attn_out, new_k, new_v = self.self_attn(norm_x, past_k=past_k, past_v=past_v)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_k, new_v


class SceneClassificationHead(nn.Module):
    """Early-layer environmental acoustic scene head (UrbanSound8K / ESC-50)."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = len(CLASSES)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, early_layer_features: torch.Tensor) -> torch.Tensor:
        # Global mean pool across temporal frames in current chunk
        pooled = torch.mean(early_layer_features, dim=1)  # (B, hidden_dim)
        return self.head(pooled)


class VoiceprintVerificationHead(nn.Module):
    """Deep-layer speaker verification head with temporal stats pooling and L2 normalization."""

    def __init__(self, hidden_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        # Stats pooling produces mean (hidden_dim) + std (hidden_dim) = 2 * hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, deep_layer_features: torch.Tensor) -> torch.Tensor:
        # deep_layer_features: (B, T_frames, hidden_dim)
        mean = torch.mean(deep_layer_features, dim=1)
        std = torch.std(deep_layer_features, dim=1)
        stats = torch.cat([mean, std], dim=-1)
        raw_emb = self.projection(stats)
        return F.normalize(raw_emb, p=2, dim=-1)


class EmotionRecognitionHead(nn.Module):
    """Deep-layer speech emotion recognition head with attentive prosody pooling."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = len(EMOTIONS)):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, deep_layer_features: torch.Tensor) -> torch.Tensor:
        # Attentive temporal pooling
        weights = F.softmax(self.attn_weights(deep_layer_features), dim=1)  # (B, T, 1)
        context = torch.sum(weights * deep_layer_features, dim=1)  # (B, hidden_dim)
        return self.classifier(context)


class MultiTaskStreamingWavLM(nn.Module):
    """Unified Edge-Optimized Multi-Task Streaming WavLM Architecture.
    
    Processes raw PCM audio chunks in real-time (< 50ms latency) using causal masked attention.
    Extracts features from early layers for environmental scene classification,
    and from deep layers for speaker voiceprints and speech emotion recognition.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 512,
        max_past_frames: int = 40,
        early_tap_layer: int = 2,
        num_scene_classes: int = len(CLASSES),
        num_emotion_classes: int = len(EMOTIONS),
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_past_frames = max_past_frames
        self.early_tap_layer = early_tap_layer

        # 1. 1D Waveform Feature Encoder
        self.feature_encoder = ConvFeatureEncoder(hidden_dim=hidden_dim)

        # 2. Causal Transformer Backbone
        self.layers = nn.ModuleList([
            StreamingWavLMBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_past_frames=max_past_frames,
            )
            for _ in range(num_layers)
        ])

        # 3. Multi-Task Heads
        self.scene_head = SceneClassificationHead(hidden_dim=hidden_dim, num_classes=num_scene_classes)
        self.voiceprint_head = VoiceprintVerificationHead(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.emotion_head = EmotionRecognitionHead(hidden_dim=hidden_dim, num_classes=num_emotion_classes)

    def forward(
        self,
        audio_chunk: torch.Tensor,
        state_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single fixed-size raw audio chunk and carry forward hidden state.
        
        Args:
            audio_chunk: (B=1, T_samples) or (B=1, 1, T_samples) raw PCM waveform
            state_cache: Optional static tensor of shape (num_layers, 2, B, num_heads, past_frames, head_dim)
                         where dimension 1 index 0 is Key, index 1 is Value.
                         
        Returns:
            scene_logits: (B, num_scene_classes)
            speaker_embedding: (B, 128) L2-normalized voiceprint
            emotion_logits: (B, num_emotion_classes)
            new_state_cache: Updated static cache tensor for the subsequent chunk.
        """
        if audio_chunk.ndim == 2:
            audio_chunk = audio_chunk.unsqueeze(1)  # (B, 1, T_samples)

        B = audio_chunk.shape[0]

        # Extract temporal frame embeddings from raw waveform
        x = self.feature_encoder(audio_chunk)  # (B, T_frames, hidden_dim)

        # Unpack state cache
        layer_k_caches: List[torch.Tensor] = []
        layer_v_caches: List[torch.Tensor] = []

        early_features: Optional[torch.Tensor] = None

        for layer_idx, block in enumerate(self.layers):
            if state_cache is not None:
                past_k = state_cache[layer_idx, 0]
                past_v = state_cache[layer_idx, 1]
            else:
                past_k = None
                past_v = None

            x, new_k, new_v = block(x, past_k=past_k, past_v=past_v)
            layer_k_caches.append(new_k)
            layer_v_caches.append(new_v)

            # Tap early layer features for environmental noise / scene sensing
            if layer_idx == self.early_tap_layer - 1:
                early_features = x

        if early_features is None:
            early_features = x

        deep_features = x  # Final layer output captures identity and semantics

        # Forward multi-task heads
        scene_logits = self.scene_head(early_features)
        speaker_embedding = self.voiceprint_head(deep_features)
        emotion_logits = self.emotion_head(deep_features)

        # Stack updated KV-caches into static tensor: (num_layers, 2, B, num_heads, past_len, head_dim)
        stacked_k = torch.stack(layer_k_caches, dim=0)
        stacked_v = torch.stack(layer_v_caches, dim=0)
        new_state_cache = torch.stack([stacked_k, stacked_v], dim=1)

        return scene_logits, speaker_embedding, emotion_logits, new_state_cache


def export_streaming_wavlm_to_onnx(
    output_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    chunk_samples: int = 3200,  # 200ms at 16kHz
    static_shapes: bool = True,
    opset_version: int = 13,
) -> Path:
    """Export MultiTaskStreamingWavLM to static-shape ONNX model optimized for TensorRT.
    
    Eliminates dynamic shapes: Traces with fixed batch size (B=1), fixed chunk length (e.g. 3200 = 200ms),
    and fixed KV-cache state shape (num_layers=6, 2, B=1, heads=4, past_frames=40, head_dim=64).
    """
    if output_path is None:
        output_path = CHECKPOINTS_DIR / "multitask_streaming_wavlm.onnx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = MultiTaskStreamingWavLM()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print(f"[WavLM Export] Loaded weights from {checkpoint_path}")
    model.eval()

    # Fixed, concrete static input shapes for maximum TensorRT optimization
    dummy_audio = torch.randn(1, chunk_samples, dtype=torch.float32)

    past_frames = 40 if static_shapes else 0
    dummy_cache = torch.zeros(6, 2, 1, 4, past_frames, 64, dtype=torch.float32)

    dynamic_axes = None
    if not static_shapes:
        dynamic_axes = {
            "state_cache_in": {4: "past_frames"},
            "state_cache_out": {4: "past_frames"},
        }

    print(
        f"[WavLM Export] Exporting MultiTaskStreamingWavLM to {output_path.name} "
        f"(Chunk: {chunk_samples} samples = {chunk_samples/16000*1000:.0f}ms, Static Shapes: {static_shapes})..."
    )

    torch.onnx.export(
        model,
        (dummy_audio, dummy_cache),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["audio_chunk", "state_cache_in"],
        output_names=["scene_logits", "speaker_embedding", "emotion_logits", "state_cache_out"],
        dynamic_axes=dynamic_axes,
    )

    filesize = output_path.stat().st_size
    print(f"[WavLM Export] Successfully generated {output_path.name} ({filesize / (1024*1024):.2f} MB, Opset {opset_version}).")
    return output_path
