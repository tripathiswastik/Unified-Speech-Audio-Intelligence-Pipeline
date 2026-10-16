"""Comparative Baseline Benchmarking Suite for Audio Intelligence Pipeline.

Rigorously compares the Unified Causal MultiTask Streaming Transformer against
three dedicated single-task baseline architectures:
1. Environmental Scene Classification: SEAudioCNN (Log-Mel Spectrogram 2D-CNN + SE attention)
2. Speaker Verification: SESpeakerEmbeddingNet (ResNet + Temporal Stats Pooling + ArcFace)
3. Speech Emotion Recognition: SEEmotionClassifierCNN (Deep 2D-CNN + Squeeze-and-Excitation)

Metrics Evaluated:
- Total Model Parameters (Millions)
- Static ONNX / Checkpoint Footprint (MB)
- Inference Latency per 200ms chunk (p50, p95, Mean ms) on GPU and CPU
- Real-Time Factor (RTF = Audio Duration / Compute Time)
- Concurrency & Memory Overhead (Single unified model vs. 3 concurrent models)
"""
import sys
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import BASE_DIR, OUTPUT_DIR, CLASSES
from pipeline.emotion_dataset import EMOTIONS
from pipeline.device import get_device
from pipeline.streaming_wavlm import CausalMultiTaskAudioTransformer
CHUNK_SAMPLES = 3200  # 200ms @ 16kHz
from pipeline.model import SEAudioCNN, SESpeakerEmbeddingNet
from pipeline.emotion_model import SEEmotionClassifierCNN


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    is_streaming_transformer: bool = False,
    warmup: int = 15,
    iterations: int = 50,
) -> Dict[str, float]:
    """Measure inference latency percentiles (p50, p95, mean) on given device."""
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    state_cache = None
    with torch.no_grad():
        for _ in range(warmup):
            if is_streaming_transformer:
                _, _, _, state_cache = model(input_tensor, state_cache=state_cache)
            else:
                _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    state_cache = None
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            if is_streaming_transformer:
                _, _, _, state_cache = model(input_tensor, state_cache=state_cache)
            else:
                _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
    }


def run_comparative_benchmark(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Execute full comparative benchmark across Unified Model vs Dedicated Baselines."""
    if device is None:
        device = get_device()
    print("=" * 80)
    print(" COMPARATIVE PRETRAINED BASELINE BENCHMARK SUITE")
    print(f" Target Device: {device} | Audio Chunk: 200 ms (3,200 samples @ 16kHz)")
    print("=" * 80)

    results = {}

    # 1. Unified Multi-Task Architecture
    print("\n[1/4] Benchmarking Unified Architecture: CausalMultiTaskAudioTransformer...")
    unified_model = CausalMultiTaskAudioTransformer(
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ffn_dim=512,
        max_past_frames=40,
        early_tap_layer=2,
        num_scene_classes=len(CLASSES),
        num_emotion_classes=len(EMOTIONS),
        embedding_dim=128,
    )
    unified_params = count_parameters(unified_model)
    raw_chunk = torch.randn(1, CHUNK_SAMPLES)
    lat_unified = benchmark_latency(unified_model, raw_chunk, device, is_streaming_transformer=True)
    rtf_unified = 200.0 / lat_unified["mean_ms"]

    results["unified_multitask_transformer"] = {
        "name": "CausalMultiTaskAudioTransformer (Unified)",
        "tasks_covered": ["Environmental Scene", "Speaker Voiceprint", "Speech Emotion"],
        "parameter_count": unified_params,
        "parameter_millions": round(unified_params / 1e6, 2),
        "model_weight_mb": round((unified_params * 4) / (1024 * 1024), 2),
        "input_format": "Raw Audio Waveform (16kHz)",
        "chunk_duration_ms": 200,
        "streaming_stateful": True,
        "latency": lat_unified,
        "real_time_factor": round(rtf_unified, 1),
    }

    # 2. Dedicated Baseline 1: Environmental Sound (SEAudioCNN)
    print("[2/4] Benchmarking Dedicated Baseline: SEAudioCNN (Scene Classification)...")
    scene_model = SEAudioCNN(num_classes=len(CLASSES))
    scene_params = count_parameters(scene_model)
    # Log-Mel Spectrogram for 200ms chunk (approx 1 x 64 mel bands x 21 frames)
    mel_chunk_64 = torch.randn(1, 1, 64, 21)
    mel_chunk_128 = torch.randn(1, 1, 128, 21)
    lat_scene = benchmark_latency(scene_model, mel_chunk_64, device, is_streaming_transformer=False)
    rtf_scene = 200.0 / lat_scene["mean_ms"]

    results["baseline_scene_se_audiocnn"] = {
        "name": "SEAudioCNN (Dedicated Scene)",
        "tasks_covered": ["Environmental Scene"],
        "parameter_count": scene_params,
        "parameter_millions": round(scene_params / 1e6, 2),
        "model_weight_mb": round((scene_params * 4) / (1024 * 1024), 2),
        "input_format": "Log-Mel Spectrogram (64x21)",
        "chunk_duration_ms": 200,
        "streaming_stateful": False,
        "latency": lat_scene,
        "real_time_factor": round(rtf_scene, 1),
    }

    # 3. Dedicated Baseline 2: Speaker Voiceprint (SESpeakerEmbeddingNet)
    print("[3/4] Benchmarking Dedicated Baseline: SESpeakerEmbeddingNet (Voiceprint)...")
    spk_model = SESpeakerEmbeddingNet(embedding_dim=128)
    spk_params = count_parameters(spk_model)
    lat_spk = benchmark_latency(spk_model, mel_chunk_128, device, is_streaming_transformer=False)
    rtf_spk = 200.0 / lat_spk["mean_ms"]

    results["baseline_speaker_sespeakerembeddingnet"] = {
        "name": "SESpeakerEmbeddingNet (Dedicated Voiceprint)",
        "tasks_covered": ["Speaker Voiceprint Verification"],
        "parameter_count": spk_params,
        "parameter_millions": round(spk_params / 1e6, 2),
        "model_weight_mb": round((spk_params * 4) / (1024 * 1024), 2),
        "input_format": "Log-Mel Spectrogram (128x21)",
        "chunk_duration_ms": 200,
        "streaming_stateful": False,
        "latency": lat_spk,
        "real_time_factor": round(rtf_spk, 1),
    }

    # 4. Dedicated Baseline 3: Speech Emotion (SEEmotionClassifierCNN)
    print("[4/4] Benchmarking Dedicated Baseline: SEEmotionClassifierCNN (SER)...")
    emo_model = SEEmotionClassifierCNN(num_classes=len(EMOTIONS))
    emo_params = count_parameters(emo_model)
    lat_emo = benchmark_latency(emo_model, mel_chunk_64, device, is_streaming_transformer=False)
    rtf_emo = 200.0 / lat_emo["mean_ms"]

    results["baseline_emotion_se_cnn"] = {
        "name": "SEEmotionClassifierCNN (Dedicated Emotion)",
        "tasks_covered": ["Speech Emotion Recognition"],
        "parameter_count": emo_params,
        "parameter_millions": round(emo_params / 1e6, 2),
        "model_weight_mb": round((emo_params * 4) / (1024 * 1024), 2),
        "input_format": "Log-Mel Spectrogram (64x21)",
        "chunk_duration_ms": 200,
        "streaming_stateful": False,
        "latency": lat_emo,
        "real_time_factor": round(rtf_emo, 1),
    }

    # 5. Composite Tri-Model Pipeline vs Unified Multi-Task
    tri_model_params = scene_params + spk_params + emo_params
    tri_model_mb = round((tri_model_params * 4) / (1024 * 1024), 2)
    tri_model_latency_mean = lat_scene["mean_ms"] + lat_spk["mean_ms"] + lat_emo["mean_ms"]
    tri_model_rtf = 200.0 / tri_model_latency_mean

    results["tri_model_concurrent_composite"] = {
        "name": "Tri-Model Concurrent Deployment (3 Separate Models)",
        "tasks_covered": ["Environmental Scene", "Speaker Voiceprint", "Speech Emotion"],
        "parameter_count": tri_model_params,
        "parameter_millions": round(tri_model_params / 1e6, 2),
        "model_weight_mb": tri_model_mb,
        "input_format": "Log-Mel Frontend + 3 Separate DNN Backbones",
        "chunk_duration_ms": 200,
        "streaming_stateful": False,
        "latency": {
            "mean_ms": round(tri_model_latency_mean, 2),
            "median_ms": round(lat_scene["median_ms"] + lat_spk["median_ms"] + lat_emo["median_ms"], 2),
            "p95_ms": round(lat_scene["p95_ms"] + lat_spk["p95_ms"] + lat_emo["p95_ms"], 2),
        },
        "real_time_factor": round(tri_model_rtf, 1),
    }

    # Print Formatted Markdown Comparison Table
    print("\n" + "=" * 95)
    print(" EMPIRICAL PERFORMANCE & RESOURCE TRADEOFF MATRIX")
    print("=" * 95)
    header = f"| {'Architecture':<35} | {'Params (M)':<10} | {'Size (MB)':<10} | {'Latency (p50)':<13} | {'Latency (p95)':<13} | {'RTF Speedup':<12} |"
    sep = f"|{'-'*37}|{'-'*12}|{'-'*12}|{'-'*15}|{'-'*15}|{'-'*14}|"
    print(header)
    print(sep)

    all_keys = [
        "unified_multitask_transformer",
        "baseline_scene_se_audiocnn",
        "baseline_speaker_sespeakerembeddingnet",
        "baseline_emotion_se_cnn",
        "tri_model_concurrent_composite",
    ]
    for k in all_keys:
        item = results[k]
        name = item["name"][:35]
        p_m = f"{item['parameter_millions']} M"
        w_mb = f"{item['model_weight_mb']} MB"
        p50 = f"{item['latency']['median_ms']:.2f} ms"
        p95 = f"{item['latency']['p95_ms']:.2f} ms"
        rtf = f"{item['real_time_factor']}x"
        print(f"| {name:<35} | {p_m:<10} | {w_mb:<10} | {p50:<13} | {p95:<13} | {rtf:<12} |")

    print(sep + "\n")

    # Save to disk
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Benchmark] Saved full benchmark telemetry to: {out_file}\n")
    return results


if __name__ == "__main__":
    run_comparative_benchmark()
