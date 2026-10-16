"""PyTorch Dataset for Speech Emotion Recognition (SER) with UrbanSound8K Noise Augmentation.
Implements Strict Speaker-Independent Disjoint Splits:
  - Train: Actors 1-18 (75%)
  - Validation: Actors 19-20 (8.3% for checkpoint selection)
  - Independent Test: Actors 21-24 (16.7% strictly held-out for unbiased benchmarking)
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, overload
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.config import BASE_DIR
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.urban_noise_mixer import UrbanNoiseMixer

RAVDESS_ROOT = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24"
RAVDESS_CSV = BASE_DIR / "pipeline" / "ravdess_metadata.csv"

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


class EmotionDataset(Dataset):
    """RAVDESS Speech Emotion Dataset with optional dynamic UrbanSound8K noise corruption."""

    def __init__(
        self,
        actors: List[int],
        metadata_csv: Path = RAVDESS_CSV,
        speech_root: Path = RAVDESS_ROOT,
        noise_mixer: Optional[UrbanNoiseMixer] = None,
        snr_range: Tuple[float, float] = (0.0, 20.0),
    ):
        self.speech_root = Path(speech_root)
        raw_df = pd.read_csv(metadata_csv)
        assert isinstance(raw_df, pd.DataFrame)
        self.df: pd.DataFrame = raw_df[raw_df["actor_id"].isin(actors)].reset_index(drop=True)
        self.noise_mixer = noise_mixer
        self.snr_range = snr_range

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        file_path = self.speech_root / row["relative_path"]

        # Load clean speech (fixed 4s window for convolutional SER model)
        waveform = load_and_preprocess_audio(file_path)
        applied_noise = "none"

        # Apply realistic UrbanSound8K environmental noise if mixer is provided
        if self.noise_mixer is not None:
            # 70% chance of noise injection during training, 30% clean
            if np.random.rand() < 0.70:
                waveform, applied_noise, _ = self.noise_mixer.mix(
                    waveform, snr_range=self.snr_range
                )

        # Compute Log-Mel Spectrogram
        spec = compute_log_mel_spectrogram(waveform)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()
        label = int(row["emotion_id"])

        return spec_tensor, label, applied_noise


@overload
def get_emotion_dataloaders(
    train_actors: List[int] = ...,
    val_actors: List[int] = ...,
    test_actors: None = None,
    batch_size: int = 32,
    use_noise_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]: ...


@overload
def get_emotion_dataloaders(
    train_actors: List[int],
    val_actors: List[int],
    test_actors: List[int],
    batch_size: int = 32,
    use_noise_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]: ...


def get_emotion_dataloaders(
    train_actors: Optional[List[int]] = None,
    val_actors: Optional[List[int]] = None,
    test_actors: Optional[List[int]] = None,
    batch_size: int = 32,
    use_noise_augmentation: bool = True,
) -> Any:
    """Create speaker-independent train, validation, and optional test DataLoaders."""
    if train_actors is None:
        train_actors = list(range(1, 19))
    if val_actors is None:
        val_actors = [19, 20]

    noise_mixer = UrbanNoiseMixer() if use_noise_augmentation else None

    train_ds = EmotionDataset(actors=train_actors, noise_mixer=noise_mixer)
    val_ds = EmotionDataset(actors=val_actors, noise_mixer=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if test_actors is not None:
        test_ds = EmotionDataset(actors=test_actors, noise_mixer=None)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
