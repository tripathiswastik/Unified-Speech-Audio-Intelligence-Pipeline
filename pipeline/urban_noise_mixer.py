"""UrbanSound8K Noise Mixer.
Dynamically injects realistic urban environmental noise into speech at configurable SNR levels.
Supports multi-clip seamless chaining and crossfading for long audio streams.
"""
from pathlib import Path
from typing import Optional, List, Tuple
import random
import numpy as np
import pandas as pd

from pipeline.config import AUDIO_DIR, METADATA_CSV, SAMPLE_RATE, TARGET_SAMPLES
from pipeline.audio_loader import load_full_audio, load_and_preprocess_audio


class UrbanNoiseMixer:
    """Blends UrbanSound8K acoustic noise into speech waveforms at target SNR (dB)."""

    def __init__(
        self,
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        classes: Optional[List[str]] = None,
        folds: Optional[List[int]] = None,
    ):
        self.audio_dir = Path(audio_dir)
        if not self.audio_dir.exists():
            raise ValueError("Audio directory folds not populated")

        if not any(self.audio_dir.glob("fold*/*.wav")):
            raise ValueError("Audio directory folds not populated")

        if not Path(metadata_csv).exists():
            raise ValueError(f"UrbanSound8K metadata not present: {metadata_csv}")

        df = pd.read_csv(metadata_csv)

        if classes:
            df = df[df["class"].isin(classes)]
        if folds:
            df = df[df["fold"].isin(folds)]

        self.noise_records = df[["slice_file_name", "fold", "class"]].to_dict("records")
        if not self.noise_records:
            raise ValueError("No UrbanSound8K noise files matched the filter criteria.")

    def __len__(self) -> int:
        return len(self.noise_records)

    def sample_noise(self, target_samples: int = TARGET_SAMPLES) -> Tuple[np.ndarray, str]:
        """Randomly sample and load UrbanSound8K noise, seamlessly chaining clips if target is long."""
        rec = random.choice(self.noise_records)
        target_class = rec["class"]
        noise_path = self.audio_dir / f"fold{rec['fold']}" / rec["slice_file_name"]

        # Use load_full_audio to preserve full uncropped noise recording, catch read errors
        try:
            waveform, _ = load_full_audio(noise_path)
        except Exception as e:
            raise ValueError(f"Audio directory folds not populated or unreadable: {noise_path}") from e

        if len(waveform) >= target_samples:
            start = random.randint(0, len(waveform) - target_samples)
            return waveform[start : start + target_samples], target_class

        # For audio longer than single noise clip, chain multiple clips from same class with crossfading
        same_class_records = [r for r in self.noise_records if r["class"] == target_class]
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
            except Exception as e:
                raise ValueError(f"Audio directory folds not populated or unreadable: {next_path}") from e
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
        return chained.astype(np.float32), target_class

    def mix(
        self,
        clean_speech: np.ndarray,
        snr_db: Optional[float] = None,
        snr_range: Tuple[float, float] = (0.0, 20.0),
    ) -> Tuple[np.ndarray, str, float]:
        """Mix clean speech with an UrbanSound8K noise clip at specified or randomized SNR.
        
        Returns:
            noisy_speech: np.ndarray (float32)
            noise_class: str
            applied_snr: float (measured empirical SNR in dB)
        """
        speech_len = len(clean_speech)
        noise_wave, noise_class = self.sample_noise(target_samples=speech_len)

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
            return clean_speech.astype(np.float32), noise_class, target_snr

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

        return mixed.astype(np.float32), noise_class, round(empirical_snr, 2)

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
        if noise_class:
            filtered = [r for r in self.noise_records if r["class"] == noise_class]
            if filtered:
                orig = self.noise_records
                self.noise_records = filtered
                try:
                    mixed, _, snr = self.mix(clean_speech, snr_db=target_snr, snr_range=snr_range)
                    return mixed, snr
                finally:
                    self.noise_records = orig
        mixed, _, snr = self.mix(clean_speech, snr_db=target_snr, snr_range=snr_range)
        return mixed, snr
