"""CUDA & Hardware Latency Profiler for Streaming WavLM.

Profiles sequential forward chunk latency on GPU (CUDA) and CPU with proper
synchronization barriers (torch.cuda.synchronize).
"""
import sys
import time
import argparse
from typing import Dict, Any
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from pipeline.streaming_wavlm import CausalMultiTaskAudioTransformer
from pipeline.config import CHECKPOINTS_DIR
from pipeline.device import safe_torch_load

CHUNK_SAMPLES = 3200  # 200ms at 16kHz
PAST_FRAMES = 40


def profile_pytorch_latency(
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_iterations: int = 100,
    warmup: int = 25,
) -> Dict[str, Any]:
    """Profile PyTorch forward pass latency across consecutive audio chunks."""
    device = torch.device(device_str)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 70)
    print(f" PYTORCH HARDWARE LATENCY BENCHMARK: {device_name} ({device})")
    print(f" Chunks: {num_iterations} iterations of {CHUNK_SAMPLES} samples (200ms)")
    print("=" * 70)

    model = CausalMultiTaskAudioTransformer().to(device)
    ckpt = CHECKPOINTS_DIR / "best_multitask_streaming_wavlm.pt"
    if ckpt.exists():
        model.load_state_dict(safe_torch_load(ckpt, map_location=device))
    model.eval()

    dummy_chunk = torch.randn(1, CHUNK_SAMPLES, device=device)
    cache = torch.zeros(6, 2, 1, 4, PAST_FRAMES, 64, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _, _, cache = model(dummy_chunk, cache)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark iterations
    latencies = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, _, cache = model(dummy_chunk, cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    arr = np.array(latencies)
    mean_val = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    rtf = 200.0 / mean_val

    print(f"  * Mean Latency:         {mean_val:.2f} ms")
    print(f"  * Median Latency (p50): {p50:.2f} ms")
    print(f"  * 90th Percentile (p90):{p90:.2f} ms")
    print(f"  * 95th Percentile (p95):{p95:.2f} ms")
    print(f"  * 99th Percentile (p99):{p99:.2f} ms")
    print(f"  * Min / Max:            {min_val:.2f} ms / {max_val:.2f} ms")
    print(f"  * Real-Time Factor:     {rtf:.1f}x real-time speedup")
    print("=" * 70)

    return {
        "device": str(device),
        "device_name": device_name,
        "mean_ms": round(mean_val, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(min_val, 2),
        "max_ms": round(max_val, 2),
        "real_time_factor": round(rtf, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Streaming WavLM hardware latency")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device: 'cuda' or 'cpu'")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    args = parser.parse_args()

    profile_pytorch_latency(device_str=args.device, num_iterations=args.iterations)


if __name__ == "__main__":
    main()
