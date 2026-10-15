"""Training & Validation Script for UrbanSound8K with Fold Isolation & Ruflo Tracking."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from pipeline.config import (
    CHECKPOINTS_DIR,
    CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEFAULT_EPOCHS,
)
from pipeline.dataset import get_dataloaders
from pipeline.model import AudioCNN
from pipeline.ruflo_tracker import RufloExperimentTracker


import time
from pipeline.device import get_device


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch [{epoch_idx:02d}/{total_epochs:02d}] Training", leave=False)
    for specs, labels in pbar:
        specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Validating", leave=False):
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
    per_class_f1_arr = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = {CLASSES[i]: float(per_class_f1_arr[i]) for i in range(len(CLASSES))}

    return avg_loss, acc, macro_f1, per_class_f1


def run_training(
    test_fold: int = 1,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
):
    device = get_device()
    print(f"\n=======================================================")
    print(f" Starting UrbanSound8K Training | Test Fold: {test_fold}")
    print(f" Target Device: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"=======================================================\n")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_dataloaders(test_fold=test_fold, batch_size=batch_size)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_checkpoint = CHECKPOINTS_DIR / f"best_model_fold{test_fold}.pt"
    tracker = RufloExperimentTracker()

    start_total_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, val_acc, val_f1, per_class_f1 = evaluate(model, val_loader, criterion, device)
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
                    "per_class_f1": per_class_f1,
                    "test_fold": test_fold,
                },
                best_checkpoint,
            )
            print(f"  --> Saved new best checkpoint to {best_checkpoint.name} (Macro-F1: {best_val_f1:.4f})")

    # Load best and log to Ruflo
    best_data = torch.load(best_checkpoint, map_location="cpu")
    tracker.log_run(
        fold=test_fold,
        epochs=epochs,
        val_accuracy=best_data["val_accuracy"],
        val_macro_f1=best_data["val_macro_f1"],
        per_class_f1=best_data["per_class_f1"],
        hyperparameters={"lr": lr, "batch_size": batch_size, "weight_decay": WEIGHT_DECAY},
        checkpoint_path=str(best_checkpoint),
    )
    print(f"\nTraining for Fold {test_fold} Completed! Best Macro-F1: {best_val_f1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSound8K Audio Classifier Training")
    parser.add_argument("--fold", type=int, default=1, help="Test fold number (1-10)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    run_training(test_fold=args.fold, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
