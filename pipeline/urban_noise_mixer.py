"""UrbanSound8K Noise Mixer.
Dynamically injects realistic urban environmental noise into speech at configurable SNR levels.
Supports multi-clip seamless chaining and crossfading for long audio streams.
"""
import os
from pathlib import Path
from typing import Optional, List, Tuple
import random
import numpy as np
import pandas as pd

from pipeline.config import AUDIO_DIR, METADATA_CSV, SAMPLE_RATE, TARGET_SAMPLES, CLASSES
from pipeline.audio_loader import load_full_audio, load_and_preprocess_audio


SUPPORTED_SYNTHETIC_CLASSES = {
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
    "white_noise",
}


def generate_synthetic_noise(noise_class: str, num_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate deterministic synthetic acoustic noise for testing/CI when audio folds are absent.
    
    Synthesizes physically plausible acoustic signatures for all 10 UrbanSound8K categories.
    Rejects unsupported noise classes rather than mislabeling arbitrary white noise.
    """
    cls = (noise_class or "white_noise").lower().strip()
    if cls not in SUPPORTED_SYNTHETIC_CLASSES:
        raise ValueError(
            f"Unsupported synthetic noise class '{noise_class}'. "
            f"Supported classes: {sorted(list(SUPPORTED_SYNTHETIC_CLASSES))}."
        )

    duration = num_samples / float(sr)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    rng = np.random.RandomState(42)

    if cls == "siren":
        # Frequency-modulated emergency siren (700Hz to 1200Hz modulated at 1.5Hz)
        f_carrier = 800.0 + 350.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f_carrier) / sr
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
        noise = 0.6 * np.sin(phase) * mod

    elif cls == "engine_idling":
        # Low frequency engine rumble + combustion harmonic pulsation
        low_tone = 0.4 * np.sin(2 * np.pi * 65.0 * t) + 0.25 * np.sin(2 * np.pi * 130.0 * t)
        flutter = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 12.0 * t))
        white = rng.normal(0, 0.15, num_samples).astype(np.float32)
        noise = (low_tone * flutter) + white

    elif cls == "air_conditioner":
        # Continuous low-frequency broadband motor hum with 120Hz/240Hz fan tones
        fan_hum = 0.3 * np.sin(2 * np.pi * 120.0 * t) + 0.15 * np.sin(2 * np.pi * 240.0 * t)
        broadband = rng.normal(0, 0.25, num_samples).astype(np.float32)
        noise = fan_hum + broadband

    elif cls == "car_horn":
        # Two-tone automotive dual-horn chord (F4 349Hz + A4 440Hz)
        horn = 0.45 * np.sin(2 * np.pi * 349.0 * t) + 0.45 * np.sin(2 * np.pi * 440.0 * t)
        horn += 0.2 * np.sin(2 * np.pi * 880.0 * t)
        # 1-second honk pulses with pauses
        gate = (np.sin(2 * np.pi * 0.8 * t) > -0.2).astype(np.float32)
        noise = horn * gate

    elif cls == "drilling":
        # Rapid rotational mechanical friction impact
        impact_train = (np.sin(2 * np.pi * 30.0 * t) > 0.1).astype(np.float32)
        whine = 0.4 * np.sin(2 * np.pi * 1200.0 * t) + 0.25 * np.sin(2 * np.pi * 2400.0 * t)
        hiss = rng.normal(0, 0.2, num_samples).astype(np.float32)
        noise = (impact_train * whine) + hiss

    elif cls == "jackhammer":
        # Heavy pneumatic percussive piston strikes at ~10 Hz with resonant ring-down
        strike_rate = 10.0
        strike_phase = (t * strike_rate) % 1.0
        strike_env = np.exp(-strike_phase * 15.0).astype(np.float32)
        strike_body = np.sin(2 * np.pi * 95.0 * t) + 0.4 * np.sin(2 * np.pi * 190.0 * t)
        clang = rng.normal(0, 0.2, num_samples).astype(np.float32)
        noise = (strike_env * strike_body) + (strike_env * clang * 0.5)

    elif cls == "gun_shot":
        # Explosive acoustic transient with sharp rise and rapid exponential decay
        shot_rate = 1.0
        shot_phase = (t * shot_rate) % 1.0
        decay = np.exp(-shot_phase * 25.0).astype(np.float32)
        blast = rng.normal(0, 0.8, num_samples).astype(np.float32) + 0.5 * np.sin(2 * np.pi * 80.0 * t)
        noise = decay * blast

    elif cls == "dog_bark":
        # Modulated downward chirp burst repeated at intervals
        bark_rate = 1.2
        bark_phase = (t * bark_rate) % 1.0
        bark_gate = (bark_phase < 0.25).astype(np.float32)
        chirp_f = 450.0 - 200.0 * (bark_phase / 0.25)
        chirp_phase = 2 * np.pi * np.cumsum(np.clip(chirp_f, 100.0, 600.0)) / sr
        noise = bark_gate * (0.6 * np.sin(chirp_phase) + rng.normal(0, 0.15, num_samples).astype(np.float32))

    elif cls == "children_playing":
        # Formant-like multi-harmonic babble and laughter bursts
        vocal1 = 0.3 * np.sin(2 * np.pi * 320.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t))
        vocal2 = 0.2 * np.sin(2 * np.pi * 640.0 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 2.8 * t))
        vocal3 = 0.15 * np.sin(2 * np.pi * 1280.0 * t)
        breath = rng.normal(0, 0.1, num_samples).astype(np.float32)
        noise = vocal1 + vocal2 + vocal3 + breath

    elif cls == "street_music":
        # Harmonic musical chord progression (C-major / G-major tones) with tempo rhythm
        chord1 = 0.3 * np.sin(2 * np.pi * 261.63 * t) + 0.25 * np.sin(2 * np.pi * 329.63 * t) + 0.2 * np.sin(2 * np.pi * 392.00 * t)
        beat = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
        noise = chord1 * beat

    else:
        # Stationary white noise (only when noise_class="white_noise")
        noise = rng.normal(0.0, 0.3, num_samples).astype(np.float32)

    return noise.astype(np.float32)


class UrbanNoiseMixer:
    """Blends UrbanSound8K acoustic noise into speech waveforms at target SNR (dB).
    
    Includes robust fallback to synthetic noise when physical dataset folds are unpopulated (e.g. CI environments).
    """

    def __init__(
        self,
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        classes: Optional[List[str]] = None,
        folds: Optional[List[int]] = None,
        allow_synthetic: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata_csv = Path(metadata_csv)
        self.classes = classes
        self.folds = folds
        self.allow_synthetic = allow_synthetic
        self.use_synthetic = False
        self.noise_records = []

        force_synthetic = os.getenv("NOISE_FORCE_SYNTHETIC", "0").strip().lower() in ("1", "true", "yes")

        has_audio = self.audio_dir.exists() and any(self.audio_dir.glob("fold*/*.wav"))
        has_meta = self.metadata_csv.exists()

        if has_audio and has_meta and not force_synthetic:
            try:
                df = pd.read_csv(self.metadata_csv)
                if classes:
                    df = df[df["class"].isin(classes)]
                if folds:
                    df = df[df["fold"].isin(folds)]
                self.noise_records = df[["slice_file_name", "fold", "class"]].to_dict("records")
                if not self.noise_records:
                    if allow_synthetic:
                        self.use_synthetic = True
                    else:
                        raise ValueError("No UrbanSound8K noise files matched the filter criteria.")
            except Exception as e:
                if allow_synthetic:
                    self.use_synthetic = True
                else:
                    raise e
        elif allow_synthetic:
            self.use_synthetic = True
        else:
            raise ValueError("Audio directory folds not populated")

    def __len__(self) -> int:
        return max(1, len(self.noise_records))

    def sample_noise(
        self,
        target_samples: int = TARGET_SAMPLES,
        target_class: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """Randomly sample and load UrbanSound8K noise, seamlessly chaining clips if target is long.
        
        Falls back to synthesized acoustic noise if audio files are unpopulated.
        """
        resolved_class = target_class
        if resolved_class is None:
            resolved_class = (
                self.classes[0]
                if (self.classes and len(self.classes) > 0)
                else (CLASSES[0] if CLASSES else "siren")
            )

        if self.use_synthetic or not self.noise_records:
            return generate_synthetic_noise(resolved_class, target_samples), resolved_class

        eligible_records = [r for r in self.noise_records if r["class"] == target_class] if target_class else self.noise_records
        if not eligible_records:
            eligible_records = self.noise_records

        rec = random.choice(eligible_records)
        chosen_class = rec["class"]
        noise_path = self.audio_dir / f"fold{rec['fold']}" / rec["slice_file_name"]

        # Use load_full_audio to preserve full uncropped noise recording, catch read errors
        try:
            waveform, _ = load_full_audio(noise_path)
        except Exception:
            if self.allow_synthetic:
                return generate_synthetic_noise(chosen_class, target_samples), chosen_class
            raise ValueError(f"Audio directory folds not populated or unreadable: {noise_path}")

        if len(waveform) >= target_samples:
            start = random.randint(0, len(waveform) - target_samples)
            return waveform[start : start + target_samples], chosen_class

        # For audio longer than single noise clip, chain multiple clips from same class with crossfading
        same_class_records = [r for r in self.noise_records if r["class"] == chosen_class]
        if not same_class_records:
            same_class_records = self.noise_records

        accumulated = [waveform]
        total_len = len(waveform)

        fade_len = int(0.05 * SAMPLE_RATE)  # 50ms crossfade
        while total_len < target_samples:
            next_rec = random.choice(same_class_records)
            next_path = self.audio_dir / f"fold{next_rec['fold']}" / next_rec["slice_file_name"]
            try:
                next_wave, _ = load_full_audio(next_path)
            except Exception:
                if self.allow_synthetic:
                    fallback_wave = generate_synthetic_noise(chosen_class, target_samples)
                    return fallback_wave, chosen_class
                raise ValueError(f"Audio directory folds not populated or unreadable: {next_path}")
            if len(next_wave) < fade_len:
                next_wave = np.pad(next_wave, (0, fade_len - len(next_wave)), mode="edge")

            # Crossfade previous tail with next head
            prev = accumulated[-1]
            if len(prev) >= fade_len:
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                blended = prev[-fade_len:] * fade_out + next_wave[:fade_len] * fade_in
                accumulated[-1] = prev[:-fade_len]
                accumulated.append(blended)
                accumulated.append(next_wave[fade_len:])
            else:
                accumulated.append(next_wave)

            total_len = sum(len(c) for c in accumulated)

        chained = np.concatenate(accumulated, axis=0)[:target_samples]
        return chained.astype(np.float32), chosen_class

    def mix(
        self,
        clean_speech: np.ndarray,
        snr_db: Optional[float] = None,
        snr_range: Tuple[float, float] = (0.0, 20.0),
        noise_class: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, float]:
        """Mix clean speech with an UrbanSound8K noise clip at specified or randomized SNR.
        
        Returns:
            noisy_speech: np.ndarray (float32)
            noise_class: str
            applied_snr: float (measured empirical SNR in dB)
        """
        speech_len = len(clean_speech)
        noise_wave, chosen_class = self.sample_noise(target_samples=speech_len, target_class=noise_class)

        # Determine target SNR
        if snr_db is None:
            target_snr = float(random.uniform(snr_range[0], snr_range[1]))
        else:
            target_snr = float(snr_db)

        # Calculate RMS powers
        speech_power = float(np.mean(clean_speech ** 2))
        noise_power = float(np.mean(noise_wave ** 2))

        if speech_power < 1e-9 or noise_power < 1e-9:
            # Silence in either speech or noise
            return clean_speech.astype(np.float32), chosen_class, target_snr

        # Calculate scale factor for noise to achieve target SNR:
        # scale = sqrt(P_speech / (P_noise * 10^(SNR/10)))
        scale = np.sqrt(speech_power / (noise_power * (10.0 ** (target_snr / 10.0))))
        scaled_noise = scale * noise_wave

        # Blend
        mixed = clean_speech + scaled_noise

        # Peak normalize to avoid clipping while preserving dynamic range
        peak = float(np.max(np.abs(mixed)))
        if peak > 1.0:
            mixed = mixed / peak

        measured_noise_power = float(np.mean(scaled_noise ** 2))
        empirical_snr = (
            float(10.0 * np.log10(speech_power / max(measured_noise_power, 1e-12)))
            if measured_noise_power > 0
            else target_snr
        )

        return mixed.astype(np.float32), chosen_class, round(empirical_snr, 2)

    def mix_noise(
        self,
        clean_speech: np.ndarray,
        noise_class: Optional[str] = None,
        target_snr: Optional[float] = None,
        snr_range: Tuple[float, float] = (0.0, 20.0),
    ) -> Tuple[np.ndarray, float]:
        """Convenience wrapper for callers expecting mix_noise(waveform, noise_class, target_snr).
        
        Returns:
            Tuple of (noisy_waveform, empirical_snr)
        """
        mixed, _, snr = self.mix(clean_speech, snr_db=target_snr, snr_range=snr_range, noise_class=noise_class)
        return mixed, snr
