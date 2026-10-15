"""UrbanSound8K Noise Mixer.
Dynamically injects realistic urban environmental noise into speech at configurable SNR levels.
"""
from pathlib import Path
from typing import Optional, List, Tuple
import random
import numpy as np
import pandas as pd

from pipeline.config import AUDIO_DIR, METADATA_CSV, SAMPLE_RATE, TARGET_SAMPLES
from pipeline.audio_loader import load_and_preprocess_audio


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
        """Randomly sample and load an UrbanSound8K noise waveform."""
        rec = random.choice(self.noise_records)
        noise_path = self.audio_dir / f"fold{rec['fold']}" / rec["slice_file_name"]
        waveform = load_and_preprocess_audio(noise_path)

        # Ensure matching length
        if len(waveform) < target_samples:
            repeats = int(np.ceil(target_samples / len(waveform)))
            waveform = np.tile(waveform, repeats)[:target_samples]
        elif len(waveform) > target_samples:
            start = random.randint(0, len(waveform) - target_samples)
            waveform = waveform[start : start + target_samples]

        return waveform, rec["class"]

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
            applied_snr: float
        """
        speech_len = len(clean_speech)
        noise_wave, noise_class = self.sample_noise(target_samples=speech_len)

        # Determine target SNR
        if snr_db is None:
            applied_snr = float(random.uniform(snr_range[0], snr_range[1]))
        else:
            applied_snr = float(snr_db)

        # Calculate RMS powers
        speech_power = np.mean(clean_speech ** 2)
        noise_power = np.mean(noise_wave ** 2)

        if speech_power < 1e-9 or noise_power < 1e-9:
            # Silence in either speech or noise
            return clean_speech.astype(np.float32), noise_class, applied_snr

        # Calculate scale factor for noise to achieve target SNR:
        # SNR = 10 * log10(P_speech / (scale^2 * P_noise))
        # scale = sqrt(P_speech / (P_noise * 10^(SNR/10)))
        scale = np.sqrt(speech_power / (noise_power * (10.0 ** (applied_snr / 10.0))))
        scaled_noise = scale * noise_wave

        # Blend
        mixed = clean_speech + scaled_noise

        # Peak normalize to avoid clipping while preserving dynamic range
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak

        return mixed.astype(np.float32), noise_class, applied_snr
