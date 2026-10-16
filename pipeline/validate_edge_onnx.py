"""Edge Deployment & Sequential Streaming Validation for ONNX Model.

Phase 5 Validation Protocol:
1. Load exported static-shape ONNX model into ONNX Runtime.
2. Feed 10 sequential 200ms audio chunks (2.0s streaming audio) with state_cache propagation.
3. Verify numerical equivalence against PyTorch model across all chunks (max error < 0.01).
4. Profile edge inference latency percentiles (p50, p95, p99, mean) over 100 sequential chunks.
5. Confirm edge latency is within real-time budget (< 50-100ms per 200ms chunk).
"""
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import CHECKPOINTS_DIR, OUTPUT_DIR, CLASSES
from pipeline.emotion_dataset import EMOTIONS
from pipeline.streaming_wavlm import (
    CausalMultiTaskAudioTransformer,
    export_streaming_wavlm_to_onnx,
)

CHUNK_SAMPLES = 3200  # 200ms at 16kHz


def validate_sequential_streaming_parity(
    onnx_path: Path,
    pytorch_ckpt_path: Path,
    num_chunks: int = 10,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Test state cache propagation over consecutive chunks comparing PyTorch vs ONNX Runtime."""
    import onnxruntime as ort

    print("=" * 80)
    print(" PHASE 5: ONNX SEQUENTIAL STREAMING & NUMERICAL PARITY VALIDATION")
    print(f" ONNX Model:     {onnx_path}")
    print(f" PyTorch Checkpoint: {pytorch_ckpt_path}")
    print(f" Chunks Tested:  {num_chunks} sequential 200ms chunks (2.0s total stream)")
    print("=" * 80)

    # 1. Initialize PyTorch model with trained weights
    pt_model = CausalMultiTaskAudioTransformer()
    pt_state = torch.load(pytorch_ckpt_path, map_location="cpu")
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    # 2. Initialize ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 3. Simulate sequential streaming audio
    past_frames = 40
    pt_cache = torch.zeros(6, 2, 1, 4, past_frames, 64, dtype=torch.float32)
    ort_cache = np.zeros((6, 2, 1, 4, past_frames, 64), dtype=np.float32)

    chunk_deltas = []

    print(f"\n{'Chunk':<8} | {'Scene Diff':<14} | {'Speaker Diff':<14} | {'Emotion Diff':<14} | {'Cache Diff':<14} | {'Status':<8}")
    print("-" * 80)

    for c_idx in range(num_chunks):
        # Deterministic pseudo-speech waveform for reproducibility
        chunk_t = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.2, CHUNK_SAMPLES, endpoint=False)).astype(np.float32)
        chunk_t += 0.2 * np.random.RandomState(42 + c_idx).randn(CHUNK_SAMPLES).astype(np.float32)
        chunk_torch = torch.from_numpy(chunk_t).unsqueeze(0)
        chunk_np = chunk_t[np.newaxis, :]

        # PyTorch forward step
        with torch.no_grad():
            pt_scene, pt_spk, pt_emo, pt_cache = pt_model(chunk_torch, pt_cache)

        # ONNX Runtime forward step
        ort_inputs = {
            "audio_chunk": chunk_np,
            "state_cache_in": ort_cache,
        }
        ort_scene, ort_spk, ort_emo, ort_cache = session.run(None, ort_inputs)

        # Compute max absolute discrepancies
        diff_scene = float(np.max(np.abs(ort_scene - pt_scene.numpy())))
        diff_spk = float(np.max(np.abs(ort_spk - pt_spk.numpy())))
        diff_emo = float(np.max(np.abs(ort_emo - pt_emo.numpy())))
        diff_cache = float(np.max(np.abs(ort_cache - pt_cache.numpy())))

        max_chunk_delta = max(diff_scene, diff_spk, diff_emo, diff_cache)
        chunk_deltas.append(max_chunk_delta)

        passed = max_chunk_delta < tolerance
        status_str = "PASS" if passed else "FAIL"
        print(f"#{c_idx+1:<7} | {diff_scene:<14.2e} | {diff_spk:<14.2e} | {diff_emo:<14.2e} | {diff_cache:<14.2e} | {status_str:<8}")

        if not passed:
            raise ValueError(f"Parity violated at chunk {c_idx+1}: delta={max_chunk_delta:.4e} > tolerance={tolerance}")

    overall_max_delta = max(chunk_deltas)
    print("-" * 80)
    print(f"[PARITY RESULT] All {num_chunks} sequential chunks matched PyTorch! Peak error: {overall_max_delta:.2e} (< {tolerance})\n")

    return {
        "num_chunks_tested": num_chunks,
        "peak_delta": overall_max_delta,
        "tolerance": tolerance,
        "parity_passed": True,
    }


def profile_edge_latency(
    onnx_path: Path,
    num_iterations: int = 100,
    warmup: int = 20,
) -> Dict[str, Any]:
    """Profile edge inference latency percentiles over sequential chunk forward passes."""
    import onnxruntime as ort

    print("=" * 80)
    print(f" EDGE HARDWARE LATENCY PROFILING (100 Iterations, 200ms audio chunks)")
    print("=" * 80)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    past_frames = 40

    dummy_chunk = np.random.randn(1, CHUNK_SAMPLES).astype(np.float32)
    state_cache = np.zeros((6, 2, 1, 4, past_frames, 64), dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        _, _, _, state_cache = session.run(None, {"audio_chunk": dummy_chunk, "state_cache_in": state_cache})

    # Timed profiling loop
    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _, _, _, state_cache = session.run(None, {"audio_chunk": dummy_chunk, "state_cache_in": state_cache})
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_val = float(np.mean(lat_arr))
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))
    rtf = 200.0 / mean_val

    meets_budget_50ms = p95 < 50.0
    meets_budget_100ms = p95 < 100.0

    print(f"  * Mean Latency:       {mean_val:.2f} ms")
    print(f"  * Median Latency (p50): {p50:.2f} ms")
    print(f"  * 90th Percentile (p90): {p90:.2f} ms")
    print(f"  * 95th Percentile (p95): {p95:.2f} ms")
    print(f"  * 99th Percentile (p99): {p99:.2f} ms")
    print(f"  * Min / Max:          {min_val:.2f} ms / {max_val:.2f} ms")
    print(f"  * Real-Time Factor:   {rtf:.1f}x real-time speedup")
    print(f"  * Sub-100ms Edge Target: {'PASSED (MEETS BUDGET)' if meets_budget_100ms else 'FAILED'}")
    print(f"  * Sub-50ms Edge Target:  {'PASSED (OPTIMAL)' if meets_budget_50ms else 'ACCEPTABLE'}")
    print("=" * 80 + "\n")

    return {
        "mean_ms": round(mean_val, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(min_val, 2),
        "max_ms": round(max_val, 2),
        "real_time_factor": round(rtf, 1),
        "meets_sub_100ms": meets_budget_100ms,
        "meets_sub_50ms": meets_budget_50ms,
    }


def run_phase_5_validation() -> Dict[str, Any]:
    """Execute end-to-end Phase 5 validation pipeline."""
    ckpt_path = CHECKPOINTS_DIR / "best_multitask_streaming_wavlm.pt"
    onnx_path = CHECKPOINTS_DIR / "multitask_streaming_wavlm.onnx"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {ckpt_path}")

    # Ensure fresh ONNX export from the latest checkpoint
    print("[Phase 5] Re-exporting static-shape ONNX from checkpoint to ensure 100% currency...")
    onnx_path = export_streaming_wavlm_to_onnx(
        output_path=onnx_path,
        checkpoint_path=ckpt_path,
        static_shapes=True,
        strict_checkpoints=True,
    )

    # Run Sequential Parity Validation
    parity_results = validate_sequential_streaming_parity(onnx_path, ckpt_path, num_chunks=10)

    # Profile Hardware Latency
    latency_results = profile_edge_latency(onnx_path, num_iterations=100)

    summary = {
        "onnx_model": str(onnx_path),
        "checkpoint": str(ckpt_path),
        "filesize_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "parity": parity_results,
        "latency": latency_results,
    }

    report_path = Path(OUTPUT_DIR) / "edge_onnx_validation_report.json"
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Phase 5] Full edge validation report written to: {report_path}\n")

    return summary


if __name__ == "__main__":
    run_phase_5_validation()
