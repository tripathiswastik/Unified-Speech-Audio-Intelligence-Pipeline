"""Training & Validation Script for Speech Emotion Recognition (SER) with Urban Noise Invariance."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from pipeline.config import CHECKPOINTS_DIR, OUTPUT_DIR
from pipeline.emotion_dataset import get_emotion_dataloaders, EMOTIONS
from pipeline.emotion_model import EmotionClassifierCNN
from pipeline.ruflo_tracker import RufloExperimentTracker


import time
from pipeline.device import get_device


def train_emotion_epoch(model, loader, criterion, optimizer, device, epoch_idx: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch [{epoch_idx:02d}/{total_epochs:02d}] Training", leave=False)
    for specs, labels, _ in pbar:
        specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


def evaluate_emotion(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for specs, labels, _ in tqdm(loader, desc="Validating", leave=False):
            specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1_arr = np.asarray(f1_score(all_targets, all_preds, average=None, zero_division=0))
    per_class_f1 = {EMOTIONS[i]: float(per_class_f1_arr[i]) for i in range(len(EMOTIONS))}

    return avg_loss, acc, macro_f1, per_class_f1


def run_emotion_training(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    use_noise_aug: bool = True,
):
    device = get_device()
    print(f"\n=======================================================")
    print(f" Starting Speech Emotion Recognition (SER) Training")
    print(f" Target Device: {device} | UrbanSound8K Noise: {use_noise_aug}")
    print(f" Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_emotion_dataloaders(
        batch_size=batch_size, use_noise_augmentation=use_noise_aug
    )

    model = EmotionClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_checkpoint = CHECKPOINTS_DIR / "best_emotion_model.pt"
    tracker = RufloExperimentTracker(experiment_name="speech_emotion_recognition_urban_augmented")

    start_total_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_emotion_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, val_acc, val_f1, per_class = evaluate_emotion(model, val_loader, criterion, device)
        scheduler.step()
        epoch_elapsed = time.time() - epoch_start
        remaining_sec = epoch_elapsed * (epochs - epoch)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s, ETA: ~{remaining_sec/60:.1f}m) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "per_class_f1": per_class,
                },
                best_checkpoint,
            )
            print(f"  --> Saved new best emotion model to {best_checkpoint.name} (F1: {val_f1:.4f})")

    print(f"\nEmotion Training Complete! Best Validation Macro-F1: {best_val_f1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-noise", action="store_true", help="Disable UrbanSound8K noise augmentation")
    args = parser.parse_args()

    run_emotion_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_noise_aug=not args.no_noise,
    )
