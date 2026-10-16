"""SampleLib Preset Benchmark WAV Generator & Audio Source Resolver.
Provides deterministic, standard-compliant 16kHz mono WAV test audio
(chirp sweeps, harmonic tones, and modulated acoustic noise) for latency,
VAD false-alarm testing, and real-time streaming simulation.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import soundfile as sf

from pipeline.config import BASE_DIR, SAMPLE_RATE, AUDIO_DIR

SAMPLE_DATA_DIR = BASE_DIR / "sample_data"


def ensure_samplelib_presets() -> Dict[str, Path]:
    """Generate and cache standardized SampleLib-compatible benchmark WAV files.
    
    Returns:
        Dict mapping descriptive preset labels to their Path on disk.
    """
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sr = SAMPLE_RATE  # 16000 Hz

    presets = {
        "SampleLib 3s Linear Chirp (VAD & Bandwidth Test)": SAMPLE_DATA_DIR / "samplelib_3s_chirp.wav",
        "SampleLib 5s Multi-Tone Harmonic (Acoustic Filter Test)": SAMPLE_DATA_DIR / "samplelib_5s_harmonic.wav",
        "SampleLib 10s Modulated Sequence (Streaming & Diarization)": SAMPLE_DATA_DIR / "samplelib_10s_modulated.wav",
    }

    # 1. 3s Linear Frequency Sweep (Chirp: 100Hz -> 4000Hz)
    p1 = presets["SampleLib 3s Linear Chirp (VAD & Bandwidth Test)"]
    if not p1.exists():
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32)
        # Chirp from 100 to 4000 Hz
        phase = 2 * np.pi * (100 * t + ((4000 - 100) / (2 * 3.0)) * (t ** 2))
        sig1 = 0.6 * np.sin(phase)
        sf.write(str(p1), sig1, sr, subtype="PCM_16")

    # 2. 5s Harmonic multi-tone (300Hz, 600Hz, 1200Hz) with 1s silence gaps for VAD testing
    p2 = presets["SampleLib 5s Multi-Tone Harmonic (Acoustic Filter Test)"]
    if not p2.exists():
        t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False, dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t) + 0.1 * np.sin(2 * np.pi * 1200 * t)
        # Apply gate: sound during [0.5s-2.0s] and [3.0s-4.5s], silence elsewhere
        gate = ((t >= 0.5) & (t <= 2.0)) | ((t >= 3.0) & (t <= 4.5))
        sig2 = (tone * gate).astype(np.float32)
        sf.write(str(p2), sig2, sr, subtype="PCM_16")

    # 3. 10s Modulated sequence simulating alternating speaker bursts and background noise
    p3 = presets["SampleLib 10s Modulated Sequence (Streaming & Diarization)"]
    if not p3.exists():
        t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False, dtype=np.float32)
        # Speaker A burst (160Hz fundamental) from 1.0s to 4.0s
        spk_a = 0.5 * np.sin(2 * np.pi * 160 * t) * ((t >= 1.0) & (t <= 4.0))
        # Speaker B burst (240Hz fundamental) from 5.5s to 8.5s
        spk_b = 0.5 * np.sin(2 * np.pi * 240 * t) * ((t >= 5.5) & (t <= 8.5))
        # Ambient low-level pinkish noise
        np.random.seed(42)
        noise = 0.02 * np.random.randn(len(t)).astype(np.float32)
        sig3 = (spk_a + spk_b + noise).astype(np.float32)
        sf.write(str(p3), sig3, sr, subtype="PCM_16")

    return presets


def get_curated_datasets() -> Dict[str, List[Tuple[str, Path]]]:
    """Return catalog of available curated dataset samples without code duplication."""
    catalog: Dict[str, List[Tuple[str, Path]]] = {}

    # RAVDESS
    rav_dir = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24"
    if rav_dir.exists():
        rav_files = list(rav_dir.glob("Actor_*/*.wav"))[:25]
        if rav_files:
            catalog["RAVDESS Speech Emotion (wo/)"] = [(f"{f.parent.name}/{f.name}", f) for f in rav_files]

    # UrbanSound8K
    if AUDIO_DIR.exists():
        urb_files = list(AUDIO_DIR.glob("fold*/*.wav"))[:25]
        if urb_files:
            catalog["UrbanSound8K Background Sound"] = [(f"{f.parent.name}/{f.name}", f) for f in urb_files]

    # VoxCeleb
    vox_dir = BASE_DIR / "wav"
    if vox_dir.exists():
        vox_files = list(vox_dir.glob("*/*/*.wav"))[:25]
        if vox_files:
            catalog["VoxCeleb Speaker (wav/)"] = [(f"{f.parent.parent.name}/{f.name}", f) for f in vox_files]

    return catalog
