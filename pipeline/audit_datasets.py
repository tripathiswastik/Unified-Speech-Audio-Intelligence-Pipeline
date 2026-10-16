"""Dataset Inventory Audit Script."""
from pathlib import Path
import pandas as pd
from pipeline.config import BASE_DIR, AUDIO_DIR, METADATA_CSV
from pipeline.emotion_dataset import RAVDESS_CSV, RAVDESS_ROOT

print("=== DATASET INVENTORY AUDIT ===")
print(f"BASE_DIR: {BASE_DIR}")

# 1. UrbanSound8K
print("\n[1] UrbanSound8K:")
print(f"  Metadata CSV: {METADATA_CSV} (exists: {METADATA_CSV.exists()})")
if METADATA_CSV.exists():
    df_u = pd.read_csv(METADATA_CSV)
    print(f"  Total metadata rows: {len(df_u)}")
print(f"  Audio Dir: {AUDIO_DIR} (exists: {AUDIO_DIR.exists()})")
if AUDIO_DIR.exists():
    folds = sorted([d.name for d in AUDIO_DIR.glob("fold*") if d.is_dir()])
    wav_count = len(list(AUDIO_DIR.glob("fold*/*.wav")))
    print(f"  Folds present: {folds}")
    print(f"  Total WAV files found: {wav_count}")

# 2. VoxCeleb
vox_root = BASE_DIR / "wav"
print("\n[2] VoxCeleb:")
print(f"  Vox Root: {vox_root} (exists: {vox_root.exists()})")
if vox_root.exists():
    spk_dirs = sorted([d.name for d in vox_root.iterdir() if d.is_dir()])
    wav_count = len(list(vox_root.glob("*/*/*.wav")))
    print(f"  Speakers found: {len(spk_dirs)} (sample: {spk_dirs[:5]})")
    print(f"  Total WAV files found: {wav_count}")

# 3. RAVDESS
print("\n[3] RAVDESS:")
print(f"  Metadata CSV: {RAVDESS_CSV} (exists: {RAVDESS_CSV.exists()})")
if RAVDESS_CSV.exists():
    df_r = pd.read_csv(RAVDESS_CSV)
    print(f"  Total metadata rows: {len(df_r)}")
print(f"  Speech Root: {RAVDESS_ROOT} (exists: {RAVDESS_ROOT.exists()})")
if RAVDESS_ROOT.exists():
    actors = sorted([d.name for d in RAVDESS_ROOT.glob("Actor_*") if d.is_dir()])
    wav_count = len(list(RAVDESS_ROOT.glob("Actor_*/*.wav")))
    print(f"  Actors present: {len(actors)}")
    print(f"  Total WAV files found: {wav_count}")
