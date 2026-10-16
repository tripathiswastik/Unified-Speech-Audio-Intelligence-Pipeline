"""Comprehensive Evaluation Script for User WAV Audio Samples.

Tests:
1. UnifiedAudioAnalyzer (Environmental Scene, Diarization, Speaker ID, SER).
2. StreamingDiarizer (Rolling real-time chunk ingestion & VAD turn segmentation).
3. Chunk-Based MultiTaskStreamingWavLM (Causal multi-task inference).
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

from pipeline.config import CLASSES, SAMPLE_RATE
from pipeline.audio_loader import load_full_audio
from pipeline.unified_analyzer import UnifiedAudioAnalyzer
from pipeline.streaming import StreamingDiarizer, StreamingPCEN
from pipeline.streaming_wavlm import MultiTaskStreamingWavLM
from pipeline.emotion_dataset import EMOTIONS


def analyze_file(file_path: Path):
    print("=" * 75)
    print(f" EVALUATING: {file_path.name}")
    print(f" Full Path:  {file_path}")
    print("=" * 75)

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    # 1. Audio Decode & Signal Characteristics
    t0 = time.time()
    waveform, duration = load_full_audio(file_path, target_sr=SAMPLE_RATE)
    decode_ms = (time.time() - t0) * 1000.0

    rms = float(np.sqrt(np.mean(waveform ** 2))) if len(waveform) > 0 else 0.0
    peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0

    print(f"\n[1. Audio Signal Characteristics]")
    print(f"  * Duration:          {duration:.2f} seconds ({len(waveform):,} samples @ {SAMPLE_RATE} Hz)")
    print(f"  * RMS Energy:        {rms:.4f} (Peak: {peak:.4f})")
    print(f"  * Decode Latency:    {decode_ms:.1f} ms")

    # 2. Unified Multi-Task Analyzer
    print(f"\n[2. Unified Audio Intelligence Analysis]")
    try:
        analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=False)
        t_start_ana = time.time()
        results = analyzer.analyze_audio_file(file_path, num_speakers=None)  # Auto-estimate speakers
        ana_time_ms = (time.time() - t_start_ana) * 1000.0

        # Environmental Scene
        env = results.get("environmental_acoustic_scene", {})
        top_cls = env.get("detected_class", "unknown")
        conf = env.get("confidence", 0.0)
        print(f"  --- Environmental Acoustic Scene (UrbanSound8K) ---")
        print(f"  * Primary Scene:     {top_cls.upper()} (Confidence: {conf*100:.1f}%)")

        # Diarization & Speech Turns
        turns = results.get("speaker_diarization_and_emotions", [])
        unique_spks = set(t.get("cluster_speaker") for t in turns if "cluster_speaker" in t)
        print(f"  --- Speaker Diarization & Emotion Recognition ---")
        print(f"  * Estimated Speakers: {len(unique_spks)} unique active speaker cluster(s) ({', '.join(sorted(unique_spks)) if unique_spks else 'None'})")
        print(f"  * Total Turns:       {len(turns)}")

        for t_idx, turn in enumerate(turns[:10]):
            spk_tag = turn.get("cluster_speaker", "Unknown")
            time_rng = turn.get("time_range", "0.00s - 0.00s")
            identified = turn.get("identified_speaker", "Unenrolled")
            spk_conf = turn.get("speaker_confidence", 0.0)
            emo = turn.get("emotion", "unknown")
            emo_conf = turn.get("emotion_confidence", 0.0)

            print(f"      Turn {t_idx+1:02d}: [{time_rng}] -> {spk_tag} (Identity: {identified} @ {spk_conf*100:.1f}%) | Emotion: {emo.upper()} ({emo_conf*100:.1f}%)")

        if len(turns) > 10:
            print(f"      ... and {len(turns) - 10} more speech turns.")

        print(f"  * Analysis Latency:  {ana_time_ms:.1f} ms ({duration / (ana_time_ms/1000):.1f}x real-time speedup)")

    except Exception as e:
        import traceback
        print(f"  Error running UnifiedAudioAnalyzer: {e}")
        traceback.print_exc()

    # 3. Streaming Diarization Simulation (250ms rolling chunks)
    print(f"\n[3. Real-Time Streaming Diarizer Simulation (250ms chunks)]")
    try:
        stream_diarizer = StreamingDiarizer(
            buffer_duration=2.0,
            hop_duration=0.5,
            sr=SAMPLE_RATE,
            energy_threshold=0.015,
        )
        chunk_ms = 250
        chunk_samples = int((chunk_ms / 1000.0) * SAMPLE_RATE)
        num_chunks = int(np.ceil(len(waveform) / chunk_samples))

        t_sim_start = time.time()
        active_chunks = 0
        window_events = 0
        speaker_shifts = 0
        prev_spk = None

        for c_idx in range(num_chunks):
            c_audio = waveform[c_idx * chunk_samples : (c_idx + 1) * chunk_samples]
            evs = stream_diarizer.push_audio(c_audio)
            rms_c = float(np.sqrt(np.mean(c_audio**2))) if len(c_audio) > 0 else 0.0
            if rms_c >= 0.015:
                active_chunks += 1
            if evs:
                for ev in evs:
                    if ev.event_type == "window":
                        window_events += 1
                        if prev_spk and prev_spk != ev.speaker_tag:
                            speaker_shifts += 1
                        prev_spk = ev.speaker_tag

        sim_dur = max(time.time() - t_sim_start, 0.0001)
        sim_speedup = duration / sim_dur
        mean_chunk_latency = (sim_dur / num_chunks) * 1000.0

        print(f"  * Processed Chunks:  {num_chunks} chunks ({chunk_ms} ms each)")
        print(f"  * Streaming Speed:   {sim_speedup:.1f}x real-time rate")
        print(f"  * Mean Chunk Time:   {mean_chunk_latency:.2f} ms / chunk")
        print(f"  * VAD Active Chunks: {active_chunks}/{num_chunks} ({(active_chunks/num_chunks)*100:.1f}%)")
        print(f"  * Speaker Shifts:    {speaker_shifts} detected handoffs")

    except Exception as e:
        print(f"  Error running StreamingDiarizer: {e}")

    # 4. Chunk-Based Streaming WavLM Multi-Task Inference (200ms chunks)
    print(f"\n[4. Chunk-Based Streaming WavLM Ingestion (200ms causal chunks)]")
    try:
        wavlm = MultiTaskStreamingWavLM().eval()
        wavlm_chunk_samples = 3200  # 200ms at 16kHz
        # Convert waveform to 16kHz for WavLM
        if SAMPLE_RATE != 16000:
            import librosa
            wav_16k = librosa.resample(waveform, orig_sr=SAMPLE_RATE, target_sr=16000)
        else:
            wav_16k = waveform

        num_wavlm_chunks = int(len(wav_16k) // wavlm_chunk_samples)
        state_cache = None
        t_wavlm_start = time.time()

        for c_idx in range(min(num_wavlm_chunks, 10)):  # sample up to 10 chunks for latency measurement
            chunk = torch.from_numpy(wav_16k[c_idx * wavlm_chunk_samples : (c_idx + 1) * wavlm_chunk_samples]).unsqueeze(0)
            with torch.no_grad():
                scene_l, spk_e, emo_l, state_cache = wavlm(chunk, state_cache=state_cache)

        t_wavlm_dur = (time.time() - t_wavlm_start) * 1000.0
        mean_wavlm_latency = t_wavlm_dur / max(1, min(num_wavlm_chunks, 10))

        # Inspect latest outputs
        pred_scene_idx = int(scene_l.argmax(dim=-1).item())
        pred_emo_idx = int(emo_l.argmax(dim=-1).item())

        print(f"  * Causal Step Time:  {mean_wavlm_latency:.2f} ms per 200ms chunk (Sub-50ms target: {'PASSED' if mean_wavlm_latency < 50 else 'EXCEEDED'})")
        print(f"  * Final State Cache: {list(state_cache.shape)} static tensor")
        print(f"  * Voiceprint Norm:   {torch.norm(spk_e, p=2).item():.4f} (Strict L2 = 1.0000)")
        print(f"  * Streaming Scene:   {CLASSES[pred_scene_idx]}")
        print(f"  * Streaming Emotion: {EMOTIONS[pred_emo_idx]}")

    except Exception as e:
        print(f"  Error running Streaming WavLM: {e}")

    print("\n" + "=" * 75 + "\n")


def main():
    f1 = Path(r"C:\Users\hp\Downloads\sample-15s.wav")
    f2 = Path(r"C:\Users\hp\Downloads\sample-speech-1m.wav")

    print("STARTING TEST RUN ON USER AUDIO SAMPLES...\n")
    analyze_file(f1)
    analyze_file(f2)


if __name__ == "__main__":
    main()
