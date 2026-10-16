"""Environment & Dataset Preflight Validator.
Audits repository structure, audio dataset presence, write permissions, and checkpoint inventory.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from pipeline.config import BASE_DIR, DATASET_DIR, AUDIO_DIR, METADATA_CSV, CHECKPOINTS_DIR, CACHE_DIR


def audit_environment() -> Dict[str, Any]:
    """Inspect environment, datasets, and checkpoints; returns a structured health audit."""
    report: Dict[str, Any] = {
        "base_dir": str(BASE_DIR),
        "exists_base_dir": BASE_DIR.exists(),
        "permissions": {},
        "datasets": {},
        "checkpoints": {},
        "status": "HEALTHY",
    }

    # 1. Directory permissions
    for name, path in [("checkpoints", CHECKPOINTS_DIR), ("cache", CACHE_DIR)]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".perm_check.tmp"
            test_file.write_text("ok")
            test_file.unlink()
            report["permissions"][name] = "WRITABLE"
        except Exception as e:
            report["permissions"][name] = f"ERROR: {e}"
            report["status"] = "WARNING"

    # 2. UrbanSound8K Dataset Audit
    urbansound_ok = True
    if not METADATA_CSV.exists():
        urbansound_ok = False
        report["datasets"]["UrbanSound8K_metadata"] = "MISSING (expected at UrbanSound8K/metadata/UrbanSound8K.csv)"
    else:
        report["datasets"]["UrbanSound8K_metadata"] = f"FOUND ({METADATA_CSV})"

    found_folds = []
    if AUDIO_DIR.exists():
        for fold in range(1, 11):
            fold_dir = AUDIO_DIR / f"fold{fold}"
            if fold_dir.exists() and len(list(fold_dir.glob("*.wav"))) > 0:
                found_folds.append(fold)
    report["datasets"]["UrbanSound8K_audio_folds"] = f"{len(found_folds)}/10 folds populated"
    if len(found_folds) < 10:
        urbansound_ok = False

    # 3. Optional Datasets
    ravdess_dir = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24"
    if ravdess_dir.exists() and len(list(ravdess_dir.glob("Actor_*/*.wav"))) > 0:
        report["datasets"]["RAVDESS_Speech"] = f"FOUND ({len(list(ravdess_dir.glob('Actor_*/*.wav')))} wav files)"
    else:
        report["datasets"]["RAVDESS_Speech"] = "NOT FOUND (optional for emotion re-training)"

    vox_dir = BASE_DIR / "wav"
    if vox_dir.exists() and len(list(vox_dir.iterdir())) > 0:
        report["datasets"]["VoxCeleb_Speakers"] = f"FOUND ({len([d for d in vox_dir.iterdir() if d.is_dir()])} speaker directories)"
    else:
        report["datasets"]["VoxCeleb_Speakers"] = "NOT FOUND (optional for speaker re-training)"

    # 4. Checkpoints Audit
    urban_checkpoints = sorted(list(CHECKPOINTS_DIR.glob("best_model_fold*.pt")))
    emotion_chk = CHECKPOINTS_DIR / "best_emotion_model.pt"
    speaker_chk = CHECKPOINTS_DIR / "best_speaker_model.pt"

    report["checkpoints"]["urban_sound_models"] = f"{len(urban_checkpoints)}/10 folds trained"
    report["checkpoints"]["emotion_model"] = "FOUND" if emotion_chk.exists() else "MISSING"
    report["checkpoints"]["speaker_model"] = "FOUND" if speaker_chk.exists() else "MISSING"

    if len(urban_checkpoints) == 0 or not emotion_chk.exists() or not speaker_chk.exists():
        if report["status"] == "HEALTHY":
            report["status"] = "INCOMPLETE_CHECKPOINTS"

    return report


def print_environment_report():
    """Print readable audit report to stdout."""
    rep = audit_environment()
    print("\n=======================================================")
    print(" [PREFLIGHT] UrbanSound Audio Intelligence Environment")
    print("=======================================================")
    print(f" Repository Root: {rep['base_dir']}")
    print(f" Overall Status:  {rep['status']}")
    print("\n--- Directory Permissions ---")
    for k, v in rep["permissions"].items():
        print(f"  [{k}] -> {v}")
    print("\n--- Dataset Corpus Audit ---")
    for k, v in rep["datasets"].items():
        print(f"  [{k}] -> {v}")
    print("\n--- Checkpoint Model Inventory ---")
    for k, v in rep["checkpoints"].items():
        print(f"  [{k}] -> {v}")
    print("=======================================================\n")

    if rep["status"] == "INCOMPLETE_CHECKPOINTS":
        print("[Guidance]: Some model checkpoints are missing. You can:")
        print("  1. Train UrbanSound8K: python -m pipeline.train --fold 1")
        print("  2. Train Emotion: python -m pipeline.train_emotion")
        print("  3. Train Speaker ID: python -m pipeline.train_speaker_id")
        print()


if __name__ == "__main__":
    print_environment_report()
