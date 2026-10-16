"""PyTorch Dataset & DataLoaders for UrbanSound8K with SpecAugment and Official Fold Split."""
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pipeline.config import AUDIO_DIR, METADATA_CSV, CACHE_DIR
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram


class SpecAugment:
    """SpecAugment data augmentation: Frequency and Time masking on Mel-spectrograms."""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 24):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        # Perform in-place masking on the caller-provided copy
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
    """UrbanSound8K Dataset partitioned strictly by official folds to prevent leakage.
    Features dual-tier bounded LRU memory cache and disk-backed .npy storage.
    """
    _LRU_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
    _MAX_LRU_ENTRIES: int = 2000  # ~170 MB max memory footprint

    def __init__(
        self,
        folds: List[int],
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        augment: bool = False,
        cache_to_disk: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        raw_df = pd.read_csv(metadata_csv)
        assert isinstance(raw_df, pd.DataFrame)
        self.df: pd.DataFrame = raw_df[raw_df["fold"].isin(folds)].reset_index(drop=True)
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None
        self.cache_to_disk = cache_to_disk
        self.disk_cache_dir = CACHE_DIR / "specs"
        if self.cache_to_disk:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        fold = int(row["fold"])
        file_name = str(row["slice_file_name"])
        cache_key = f"{fold}_{file_name}"

        # 1. Tier 1: Check Bounded In-Memory LRU Cache
        if cache_key in self._LRU_CACHE:
            spec = self._LRU_CACHE[cache_key]
            self._LRU_CACHE.move_to_end(cache_key)
        else:
            # 2. Tier 2: Check Disk-Backed .npy Cache
            disk_path = self.disk_cache_dir / f"{cache_key}.npy"
            if self.cache_to_disk and disk_path.exists():
                try:
                    spec = np.load(str(disk_path))
                except Exception:
                    spec = None
            else:
                spec = None

            if spec is None:
                file_path = self.audio_dir / f"fold{fold}" / file_name
                waveform = load_and_preprocess_audio(file_path)
                spec = compute_log_mel_spectrogram(waveform)
                if self.cache_to_disk:
                    try:
                        np.save(str(disk_path), spec)
                    except Exception:
                        pass

            # Store in bounded LRU cache
            if len(self._LRU_CACHE) >= self._MAX_LRU_ENTRIES:
                self._LRU_CACHE.popitem(last=False)
            self._LRU_CACHE[cache_key] = spec

        # Data augmentation (single copy only when augment is active)
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
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders using official 10-fold split with memory pinning."""
    train_folds = [f for f in range(1, 11) if f != test_fold]
    val_folds = [test_fold]

    train_ds = UrbanSoundDataset(folds=train_folds, augment=True)
    val_ds = UrbanSoundDataset(folds=val_folds, augment=False)

    use_pin = pin_memory if pin_memory is not None else torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
