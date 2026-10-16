"""PyTorch Dataset & DataLoaders for UrbanSound8K with SpecAugment and Official Fold Split."""
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.config import AUDIO_DIR, METADATA_CSV
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram


class SpecAugment:
    """SpecAugment data augmentation: Frequency and Time masking on Mel-spectrograms."""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 24):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()
        n_mels, n_time = spec.shape

        # Frequency mask
        f = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        spec[f0 : f0 + f, :] = 0.0

        # Time mask
        t = np.random.randint(0, self.time_mask_param)
        t0 = np.random.randint(0, n_time - t)
        spec[:, t0 : t0 + t] = 0.0

        return spec


class UrbanSoundDataset(Dataset):
    """UrbanSound8K Dataset partitioned strictly by official folds to prevent leakage."""
    _SPEC_CACHE: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        folds: List[int],
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        augment: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        raw_df = pd.read_csv(metadata_csv)
        assert isinstance(raw_df, pd.DataFrame)
        self.df: pd.DataFrame = raw_df[raw_df["fold"].isin(folds)].reset_index(drop=True)
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        file_name = str(row["slice_file_name"])

        if file_name in self._SPEC_CACHE:
            spec = self._SPEC_CACHE[file_name]
        else:
            file_path = self.audio_dir / f"fold{row['fold']}" / file_name
            waveform = load_and_preprocess_audio(file_path)
            spec = compute_log_mel_spectrogram(waveform)
            self._SPEC_CACHE[file_name] = spec

        if self.augment and self.spec_augment is not None:
            spec = self.spec_augment(spec.copy())

        # Add channel dimension: (1, n_mels, time_steps)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()
        label = int(row["classID"])
        return spec_tensor, label


def get_dataloaders(
    test_fold: int,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders using official 10-fold split.
    Validation is strictly `test_fold`, train is the remaining 9 folds.
    """
    train_folds = [f for f in range(1, 11) if f != test_fold]
    val_folds = [test_fold]

    train_ds = UrbanSoundDataset(folds=train_folds, augment=True)
    val_ds = UrbanSoundDataset(folds=val_folds, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
