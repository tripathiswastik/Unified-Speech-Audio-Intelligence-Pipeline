"""Training & Validation Script for VoxCeleb Speaker Embedding Network.
Implements Normalized Cosine Cross-Entropy over VoxCeleb speaker identities to produce
fine-tuned 128-dimensional L2-normalized voiceprint embeddings.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pipeline.config import BASE_DIR, CHECKPOINTS_DIR
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.speaker_id import SpeakerEmbeddingNet
from pipeline.dataset import SpecAugment
from pipeline.device import get_device
from pipeline.ruflo_tracker import RufloExperimentTracker


class VoxCelebDataset(Dataset):
    """Dataset for VoxCeleb audio files with in-memory spectrogram caching."""
    _CACHE: Dict[str, np.ndarray] = {}

    def __init__(self, file_paths: List[Path], labels: List[int], augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        fp = self.file_paths[index]
        key = str(fp)
        if key in self._CACHE:
            spec = self._CACHE[key]
        else:
            waveform = load_and_preprocess_audio(fp)
            spec = compute_log_mel_spectrogram(waveform)
            self._CACHE[key] = spec

        if self.augment and self.spec_augment is not None:
            spec = self.spec_augment(spec.copy())

        tensor = torch.from_numpy(spec).unsqueeze(0).float()
        label = self.labels[index]
        return tensor, label


class SpeakerCosineClassifier(nn.Module):
    """Wrapper that combines the 128-d SpeakerEmbeddingNet with a normalized Cosine Linear head."""

    def __init__(self, embedding_net: SpeakerEmbeddingNet, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.embedding_net = embedding_net
        self.scale = scale
        self.classifier = nn.Linear(128, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding_net(x)
        # Normalize classifier weights so dot product equals cosine similarity
        weight_norm = F.normalize(self.classifier.weight, p=2, dim=-1)
        logits = F.linear(emb, weight_norm) * self.scale
        return logits, emb


def get_voxceleb_dataloaders(
    vox_root: Optional[Path] = None,
    max_speakers: int = 50,
    max_files_per_speaker: int = 50,
    val_ratio: float = 0.15,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """Scan VoxCeleb directory, create speaker label mapping, and construct train/val loaders."""
    if vox_root is None:
        vox_root = BASE_DIR / "wav"

    all_speaker_dirs = sorted([d for d in vox_root.iterdir() if d.is_dir()])
    if max_speakers > 0 and len(all_speaker_dirs) > max_speakers:
        selected_dirs = all_speaker_dirs[:max_speakers]
    else:
        selected_dirs = all_speaker_dirs

    speaker_names = [d.name for d in selected_dirs]
    spk_to_id = {name: idx for idx, name in enumerate(speaker_names)}

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    np.random.seed(42)
    for spk_dir in selected_dirs:
        spk_id = spk_to_id[spk_dir.name]
        files = sorted(list(spk_dir.glob("*/*.wav")))
        if not files:
            continue

        if max_files_per_speaker > 0 and len(files) > max_files_per_speaker:
            files = files[:max_files_per_speaker]

        n_val = max(1, int(len(files) * val_ratio))
        np.random.shuffle(files)
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            train_paths.append(f)
            train_labels.append(spk_id)

        for f in val_files:
            val_paths.append(f)
            val_labels.append(spk_id)

    train_ds = VoxCelebDataset(train_paths, train_labels, augment=True)
    val_ds = VoxCelebDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[VoxCeleb Loader] Loaded {len(speaker_names)} speakers | Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader, len(speaker_names), speaker_names


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch [{epoch_idx:02d}/{total_epochs:02d}] Training", leave=False)

    for specs, labels in pbar:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader.dataset))


def evaluate_speaker_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Validating", leave=False):
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            _, preds_top1 = logits.max(dim=1)
            correct_top1 += int((preds_top1 == labels).sum().item())

            if logits.size(1) >= 5:
                _, preds_top5 = logits.topk(5, dim=1)
                correct_top5 += int(preds_top5.eq(labels.view(-1, 1).expand_as(preds_top5)).sum().item())
            else:
                correct_top5 = correct_top1

            total_samples += len(labels)

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = correct_top1 / max(1, total_samples)
    top5_acc = correct_top5 / max(1, total_samples)
    return avg_loss, top1_acc, top5_acc


def run_speaker_training(
    max_speakers: int = 50,
    max_files_per_speaker: int = 50,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    device = get_device()
    print(f"\n=======================================================")
    print(f" Starting VoxCeleb Speaker Embedding Network Training")
    print(f" Target Device: {device} | Max Speakers: {max_speakers} | Max Files/Spk: {max_files_per_speaker} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"=======================================================\n")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, num_classes, speaker_names = get_voxceleb_dataloaders(
        max_speakers=max_speakers, max_files_per_speaker=max_files_per_speaker, batch_size=batch_size
    )

    embedding_net = SpeakerEmbeddingNet().to(device)
    model = SpeakerCosineClassifier(embedding_net, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_top1_acc = 0.0
    best_checkpoint = CHECKPOINTS_DIR / "best_speaker_model.pt"
    tracker = RufloExperimentTracker(experiment_name="voxceleb_speaker_embedding")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, top1_acc, top5_acc = evaluate_speaker_model(model, val_loader, criterion, device)
        scheduler.step()
        epoch_elapsed = time.time() - epoch_start
        remaining_sec = epoch_elapsed * (epochs - epoch)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s, ETA: ~{remaining_sec/60:.1f}m) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1 Acc: {top1_acc*100:.2f}% | "
            f"Top-5 Acc: {top5_acc*100:.2f}%"
        )

        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            # Save the underlying SpeakerEmbeddingNet weights alone for clean inference export
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": embedding_net.state_dict(),
                    "top1_accuracy": top1_acc,
                    "top5_accuracy": top5_acc,
                    "num_speakers": num_classes,
                    "speaker_names": speaker_names,
                },
                best_checkpoint,
            )
            print(f"  --> Saved new best speaker checkpoint to {best_checkpoint.name} (Top-1: {best_top1_acc*100:.2f}%)")

    # Record training run in Ruflo tracker
    tracker.log_run(
        fold=0,
        epochs=epochs,
        val_accuracy=best_top1_acc,
        val_macro_f1=best_top1_acc,
        per_class_f1={"top1_accuracy": best_top1_acc},
        hyperparameters={"max_speakers": num_classes, "max_files_per_speaker": max_files_per_speaker, "lr": lr, "batch_size": batch_size},
        checkpoint_path=str(best_checkpoint),
    )

    print(f"\nTraining Completed! Best Top-1 Accuracy: {best_top1_acc*100:.2f}%\n")
    return {
        "best_top1_accuracy": best_top1_acc,
        "checkpoint": str(best_checkpoint),
        "num_speakers": num_classes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCeleb Speaker Embedding Network Training")
    parser.add_argument("--max-speakers", type=int, default=50, help="Maximum number of speakers to train on (0 for all)")
    parser.add_argument("--max-files-per-speaker", type=int, default=50, help="Maximum audio files per speaker")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    run_speaker_training(
        max_speakers=args.max_speakers,
        max_files_per_speaker=args.max_files_per_speaker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
