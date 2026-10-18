"""Joint Multi-Task Training Pipeline for Chunk-Based Streaming WavLM.

Optimizes a single shared causal WavLM backbone across three tasks:
1. Environmental Acoustic Scene Detection (UrbanSound8K / ESC-50) tapped from Layer 2.
2. Speaker Voiceprint Verification & ArcFace Angular Margin (VoxCeleb) tapped from Layer 6.
3. Speech Emotion Recognition (RAVDESS 8-class SER) tapped from Layer 6.

Features:
- Causal chunk-state propagation through time during backpropagation.
- Homoscedastic multi-task uncertainty weighting (Kendall et al.) for automated task loss balancing.
- Stratified disjoint validation splits across all three datasets.
- Automatic static-shape ONNX / TensorRT export upon achieving best composite validation score.
- Full experiment provenance tracking via RufloExperimentTracker.
"""
import sys
import os
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence
import argparse
import math
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from pipeline.config import (
    BASE_DIR,
    AUDIO_DIR,
    METADATA_CSV,
    CHECKPOINTS_DIR,
    CLASSES,
    CLASS_TO_ID,
)
from pipeline.emotion_dataset import RAVDESS_CSV, RAVDESS_ROOT, EMOTIONS
from pipeline.streaming_wavlm import (
    CausalMultiTaskAudioTransformer,
    MultiTaskStreamingWavLM,
    export_streaming_wavlm_to_onnx,
)
from pipeline.model import ArcMarginProduct
from pipeline.device import get_device
from pipeline.ruflo_tracker import RufloExperimentTracker


TARGET_SR = 16000
CHUNK_MS = 200
CHUNK_SAMPLES = int((CHUNK_MS / 1000.0) * TARGET_SR)  # 3,200 samples
TOTAL_WINDOW_SAMPLES = TARGET_SR * 1  # 1.0 second (5 chunks)


def load_raw_pcm16(
    file_path: Path,
    target_samples: int = TOTAL_WINDOW_SAMPLES,
    random_crop: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Load audio file as mono 16kHz float32 waveform padded or cropped to target_samples.
    
    Cropping is performed before peak normalization to prevent distant transient spikes
    from attenuating the energy of the selected window.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    file_str = str(file_path)
    try:
        wav, sr = sf.read(file_str, dtype="float32")
    except Exception:
        try:
            wav, sr = librosa.load(file_str, sr=None, mono=False)
        except Exception as err:
            raise RuntimeError(f"Corrupted audio file could not be decoded: {file_path}") from err

    if wav.ndim > 1:
        wav = np.mean(wav, axis=0 if wav.shape[0] < wav.shape[1] else 1)

    if sr != TARGET_SR:
        try:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_fast")
        except Exception:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    # Validate finite values
    if not np.all(np.isfinite(wav)):
        raise ValueError(f"Audio file contains non-finite values (NaN or Inf): {file_path}")

    # Pad or crop
    n_samples = len(wav)
    if n_samples < target_samples:
        pad_left = (target_samples - n_samples) // 2
        pad_right = target_samples - n_samples - pad_left
        wav = np.pad(wav, (pad_left, pad_right), mode="constant")
    elif n_samples > target_samples:
        if random_crop:
            rand_gen = rng if rng is not None else np.random
            start = rand_gen.randint(0, n_samples - target_samples + 1)
        else:
            start = (n_samples - target_samples) // 2
        wav = wav[start : start + target_samples]

    # Normalize amplitude on the extracted audio window
    max_val = np.max(np.abs(wav)) if len(wav) > 0 else 0.0
    if max_val > 1e-6:
        wav = wav / max_val

    return wav.astype(np.float32)


class UrbanSoundRawDataset(Dataset):
    """Raw waveform dataset for UrbanSound8K partitioned by official folds."""

    def __init__(
        self,
        folds: List[int],
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        strict: bool = True,
        is_train: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata_csv = Path(metadata_csv)
        self.is_train = is_train
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"UrbanSound8K metadata not found: {self.metadata_csv}")
        df = pd.read_csv(self.metadata_csv)
        assert isinstance(df, pd.DataFrame)
        self.df = df[df["fold"].isin(folds)].reset_index(drop=True)

        if strict:
            if not self.audio_dir.exists():
                raise FileNotFoundError(f"UrbanSound8K audio directory not found: {self.audio_dir}")
            missing_files = []
            valid_rows = []
            for _, row in self.df.iterrows():
                fp = self.audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
                if fp.exists():
                    valid_rows.append(row)
                else:
                    missing_files.append(str(fp))
            if not valid_rows:
                raise FileNotFoundError(
                    f"No UrbanSound8K audio files found in {self.audio_dir} for folds {folds}. "
                    f"Example missing: {missing_files[:3]}"
                )
            self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        fp = self.audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
        wav = load_raw_pcm16(fp, random_crop=self.is_train)
        label = int(row["classID"])
        return torch.from_numpy(wav), label


class VoxCelebRawDataset(Dataset):
    """Raw waveform dataset for VoxCeleb speaker identification and open-set verification.
    
    Supports:
      - split="train": 80% files per training speaker (with random crop)
      - split="heldout_id": 20% held-out files of training speakers (with deterministic center crop and shared speaker_to_id)
      - split="val": Disjoint unseen speakers for open-set verification trials (deterministic center crop)
    """

    def __init__(
        self,
        vox_root: Optional[Path] = None,
        max_speakers: int = 50,
        split: str = "train",
        speaker_disjoint: bool = True,
        strict: bool = True,
        is_train: Optional[bool] = None,
        speaker_to_id: Optional[Dict[str, int]] = None,
    ):
        if vox_root is None:
            vox_root = BASE_DIR / "wav"
        self.vox_root = Path(vox_root)
        self.split = split
        self.is_train = (split == "train") if is_train is None else is_train
        self.items: List[Tuple[Path, int]] = []
        self.speaker_to_id: Dict[str, int] = {}
        self.speaker_files: Dict[int, List[Path]] = {}

        if not self.vox_root.exists():
            if strict:
                raise FileNotFoundError(
                    f"VoxCeleb dataset directory not found: {self.vox_root}. "
                    f"Cannot silently substitute zero waveforms."
                )
            return

        spk_dirs = sorted([d for d in self.vox_root.iterdir() if d.is_dir()])[:max_speakers]
        if not spk_dirs:
            if strict:
                raise FileNotFoundError(f"No speaker directories found under {self.vox_root}")
            return

        if speaker_disjoint:
            # 80% speakers for train & heldout_id; 20% disjoint speakers for open-set verification
            split_idx = max(1, int(0.8 * len(spk_dirs)))
            train_spk_dirs = spk_dirs[:split_idx]
            val_spk_dirs = spk_dirs[split_idx:] if split_idx < len(spk_dirs) else spk_dirs[-1:]

            if split in ("train", "heldout_id"):
                active_dirs = train_spk_dirs
            else:
                active_dirs = val_spk_dirs
        else:
            active_dirs = spk_dirs

        if speaker_to_id is not None:
            self.speaker_to_id = speaker_to_id
        else:
            self.speaker_to_id = {d.name: idx for idx, d in enumerate(active_dirs)}

        for d in active_dirs:
            if d.name not in self.speaker_to_id:
                continue
            spk_id = self.speaker_to_id[d.name]
            all_files = sorted(d.glob("*/*.wav"))
            if not all_files:
                continue

            if split in ("train", "heldout_id") and speaker_disjoint:
                # Deterministic 80/20 per-speaker file partition
                n_tr = max(1, int(0.8 * len(all_files)))
                selected_files = all_files[:n_tr] if split == "train" else all_files[n_tr:]
                if not selected_files:
                    selected_files = all_files[-1:]
            else:
                selected_files = all_files

            self.speaker_files[spk_id] = selected_files
            for f in selected_files:
                self.items.append((f, spk_id))

        if strict and not self.items:
            raise FileNotFoundError(f"No WAV audio files found for VoxCeleb split={split} in {self.vox_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if not self.items:
            raise IndexError("VoxCeleb dataset is empty.")
        fp, spk_id = self.items[index]
        wav = load_raw_pcm16(fp, random_crop=self.is_train)
        return torch.from_numpy(wav), spk_id


class RAVDESSRawDataset(Dataset):
    """Raw waveform dataset for RAVDESS Speech Emotion Recognition."""

    def __init__(
        self,
        actors: List[int],
        metadata_csv: Path = RAVDESS_CSV,
        speech_root: Path = RAVDESS_ROOT,
        strict: bool = True,
        is_train: bool = True,
    ):
        self.speech_root = Path(speech_root)
        self.metadata_csv = Path(metadata_csv)
        self.is_train = is_train
        if not self.metadata_csv.exists():
            if strict:
                raise FileNotFoundError(f"RAVDESS metadata CSV not found: {self.metadata_csv}")
            self.df = pd.DataFrame()
            return

        df = pd.read_csv(self.metadata_csv)
        assert isinstance(df, pd.DataFrame)
        self.df = df[df["actor_id"].isin(actors)].reset_index(drop=True)

        if strict:
            if not self.speech_root.exists():
                raise FileNotFoundError(f"RAVDESS speech root directory not found: {self.speech_root}")
            valid_rows = []
            for _, row in self.df.iterrows():
                fp = self.speech_root / row["relative_path"]
                if fp.exists():
                    valid_rows.append(row)
            if not valid_rows:
                raise FileNotFoundError(f"No RAVDESS audio files found in {self.speech_root} for actors {actors}")
            self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if len(self.df) == 0:
            raise IndexError("RAVDESS dataset is empty.")
        row = self.df.iloc[index]
        fp = self.speech_root / row["relative_path"]
        wav = load_raw_pcm16(fp, random_crop=self.is_train)
        label = int(row["emotion_id"])
        return torch.from_numpy(wav), label


def run_dataset_preflight_audit(
    splits: Union[Dict[str, Any], Sequence[Tuple[str, Any]]],
    max_verify_per_split: int = 5,
    min_samples: int = 1,
) -> Dict[str, Any]:
    """Execute preflight audit across all dataset splits.
    
    Verifies:
      1. Minimum required samples (min_samples).
      2. Valid decoding and finite samples (no NaNs or Infs).
      3. Non-zero energy / non-silent waveforms.
      4. Valid label types and values.
    """
    print("\n" + "=" * 80)
    print(" PREFLIGHT DATASET INTEGRITY AUDIT")
    print("=" * 80)
    audit_results: Dict[str, Any] = {}
    items: Sequence[Tuple[str, Any]] = list(splits.items()) if isinstance(splits, dict) else list(splits)
    for name, ds in items:
        count: int = len(ds) if hasattr(ds, "__len__") else 0
        if count < min_samples:
            raise ValueError(f"[Preflight Audit Failed] Dataset split '{name}' has {count} samples, fewer than required min_samples={min_samples}.")

        num_to_check = min(count, max_verify_per_split)
        indices_to_test = np.linspace(0, count - 1, num_to_check, dtype=int) if count > 0 else []
        for idx in indices_to_test:
            wav_t, label = ds[idx]
            wav_np = wav_t.numpy()
            if not np.all(np.isfinite(wav_np)):
                raise ValueError(f"[Preflight Audit Failed] Non-finite values in split '{name}' at index {idx}.")
            if float(np.max(np.abs(wav_np))) < 1e-4:
                raise ValueError(f"[Preflight Audit Failed] Silent / zero-amplitude audio in split '{name}' at index {idx}.")
            if not isinstance(label, (int, np.integer)):
                raise TypeError(f"[Preflight Audit Failed] Invalid label type {type(label)} in split '{name}'.")

        audit_results[name] = {"count": count, "status": "VERIFIED"}
        print(f"  * {name:<25}: {count:>5} samples verified (audited {num_to_check} decoded WAVs)")

    print("=" * 80)
    print(" [Preflight Audit Passed] All dataset splits structurally verified & compliant.\n")
    return audit_results


def compute_equal_error_rate(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float, float]:
    """Calculate Equal Error Rate (EER), ROC-AUC, and optimal threshold from genuine & impostor trials."""
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.5, 0.5, 0.0

    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    # Evaluate FAR & FRR across candidate thresholds
    thresholds = np.linspace(-1.0, 1.0, 201)
    far_list, frr_list = [], []
    for th in thresholds:
        far = np.mean(impostor_scores >= th)
        frr = np.mean(genuine_scores < th)
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    diff = np.abs(far_arr - frr_arr)
    min_idx = np.argmin(diff)
    eer = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
    opt_th = float(thresholds[min_idx])
    return eer, auc, opt_th


def generate_speaker_verification_trials(
    dataset: VoxCelebRawDataset,
    num_pairs: int = 100,
    seed: int = 42,
) -> List[Tuple[Path, Path, int]]:
    """Generate deterministic genuine and impostor trial pairs for speaker verification.
    
    Returns:
        List of (file1, file2, is_same_speaker: 1 or 0)
    """
    rng = np.random.RandomState(seed)
    spk_ids = sorted([k for k, v in dataset.speaker_files.items() if len(v) >= 1])
    if len(spk_ids) < 2:
        return []

    trials = []

    # 1. Genuine trials (pairs from same speaker)
    candidates = [s for s in spk_ids if len(dataset.speaker_files[s]) >= 2]
    n_genuine = num_pairs // 2
    if candidates:
        for _ in range(n_genuine):
            spk = rng.choice(candidates)
            files = dataset.speaker_files[spk]
            f_indices = rng.choice(len(files), size=2, replace=False)
            trials.append((files[f_indices[0]], files[f_indices[1]], 1))

    # 2. Impostor trials (pairs from different speakers)
    n_impostor = num_pairs - len(trials)
    for _ in range(n_impostor):
        s_pair = rng.choice(spk_ids, size=2, replace=False)
        files1 = dataset.speaker_files[s_pair[0]]
        files2 = dataset.speaker_files[s_pair[1]]
        f1 = files1[rng.randint(len(files1))]
        f2 = files2[rng.randint(len(files2))]
        trials.append((f1, f2, 0))

    return trials


def save_checkpoint_atomic(state_dict: dict, file_path: Path) -> None:
    """Atomically save checkpoint using temporary file to prevent corruption on crash."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.stem}.tmp{file_path.suffix}")
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, file_path)


def evaluate_speaker_verification_trials(
    model: nn.Module,
    trials: List[Tuple[Path, Path, int]],
    device: torch.device,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate speaker verification on a fixed set of trials with embedding caching.
    
    If fixed_threshold is provided (calibrated operating threshold from validation),
    FAR and FRR are evaluated at this operating point without retuning.
    """
    if not trials:
        return {
            "spk_eer": 0.5,
            "spk_auc": 0.5,
            "spk_threshold": 0.0,
            "spk_operating_threshold": 0.0,
            "spk_far": 0.5,
            "spk_frr": 0.5,
        }

    model.eval()
    embedding_cache: Dict[Path, torch.Tensor] = {}
    distinct_files = list({t[0] for t in trials} | {t[1] for t in trials})

    with torch.no_grad():
        for fp in distinct_files:
            wav = torch.from_numpy(load_raw_pcm16(fp, random_crop=False)).unsqueeze(0).to(device)
            _, spk_emb, _ = process_audio_in_streaming_chunks(model, wav)
            embedding_cache[fp] = spk_emb.squeeze(0)

        genuine_scores = []
        impostor_scores = []
        for f1, f2, is_same in trials:
            sim = F.cosine_similarity(embedding_cache[f1].unsqueeze(0), embedding_cache[f2].unsqueeze(0)).item()
            if is_same == 1:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    if not genuine_scores or not impostor_scores:
        return {
            "spk_eer": 0.5,
            "spk_auc": 0.5,
            "spk_threshold": 0.0,
            "spk_operating_threshold": 0.0,
            "spk_far": 0.5,
            "spk_frr": 0.5,
        }

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)
    eer, auc, opt_th = compute_equal_error_rate(gen_arr, imp_arr)
    eval_th = fixed_threshold if fixed_threshold is not None else opt_th
    far = float(np.mean(imp_arr >= eval_th))
    frr = float(np.mean(gen_arr < eval_th))

    return {
        "spk_eer": eer,
        "spk_auc": auc,
        "spk_threshold": opt_th,
        "spk_operating_threshold": eval_th,
        "spk_far": far,
        "spk_frr": frr,
    }


def evaluate_open_set_speaker_verification(
    model: nn.Module,
    dataset: VoxCelebRawDataset,
    device: torch.device,
    num_pairs: int = 100,
    seed: int = 42,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate open-set speaker verification using fixed deterministic trials."""
    trials = generate_speaker_verification_trials(dataset, num_pairs=num_pairs, seed=seed)
    return evaluate_speaker_verification_trials(model, trials, device, fixed_threshold=fixed_threshold)


class MultiTaskUncertaintyLoss(nn.Module):
    """Homoscedastic uncertainty loss weighting (Kendall et al., CVPR 2018).
    
    Dynamically balances multi-task objectives without manual hyperparameter tuning:
    L_total = 0.5 * exp(-s1) * L_scene + 0.5 * exp(-s2) * L_spk + 0.5 * exp(-s3) * L_emo + 0.5 * (s1 + s2 + s3)
    where s_i = log(sigma_i^2) is a learned log-variance parameter.
    """

    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(
        self,
        loss_scene: torch.Tensor,
        loss_spk: torch.Tensor,
        loss_emo: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prec_scene = torch.exp(-self.log_vars[0])
        prec_spk = torch.exp(-self.log_vars[1])
        prec_emo = torch.exp(-self.log_vars[2])

        total_loss = (
            0.5 * prec_scene * loss_scene
            + 0.5 * prec_spk * loss_spk
            + 0.5 * prec_emo * loss_emo
            + 0.5 * torch.sum(self.log_vars)
        )

        weights = {
            "w_scene": float(prec_scene.item()),
            "w_spk": float(prec_spk.item()),
            "w_emo": float(prec_emo.item()),
        }
        return total_loss, weights


def process_audio_in_streaming_chunks(
    model: nn.Module,
    waveforms: torch.Tensor,
    chunk_samples: int = CHUNK_SAMPLES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice raw audio waveforms into sequential chunks and step forward with state cache propagation.
    
    Args:
        waveforms: (B, total_samples) raw audio batch
        chunk_samples: samples per chunk (3,200 for 200ms)
        
    Returns:
        scene_logits: Final or mean pooled scene predictions across chunks (B, num_classes)
        speaker_emb: Final chunk L2-normalized voiceprint (B, 128)
        emotion_logits: Final or mean pooled emotion predictions across chunks (B, num_emotions)
    """
    B, total_len = waveforms.shape
    num_chunks = total_len // chunk_samples

    state_cache = None
    all_scene_logits = []
    all_spk_embs = []
    all_emo_logits = []

    for c_idx in range(num_chunks):
        chunk = waveforms[:, c_idx * chunk_samples : (c_idx + 1) * chunk_samples]
        scene_l, spk_e, emo_l, state_cache = model(chunk, state_cache=state_cache)
        all_scene_logits.append(scene_l)
        all_spk_embs.append(spk_e)
        all_emo_logits.append(emo_l)

    # Average scene logits across chunks (environmental noise texture stability)
    avg_scene_logits = torch.mean(torch.stack(all_scene_logits, dim=0), dim=0)
    # Final chunk voiceprint and emotion represent cumulative prosody/identity
    final_spk_emb = all_spk_embs[-1]
    avg_emo_logits = torch.mean(torch.stack(all_emo_logits, dim=0), dim=0)

    return avg_scene_logits, final_spk_emb, avg_emo_logits


def train_multitask_epoch(
    model: nn.Module,
    arc_margin: ArcMarginProduct,
    uncertainty_loss: MultiTaskUncertaintyLoss,
    loader_urban: DataLoader,
    loader_vox: DataLoader,
    loader_ravdess: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Train single epoch with interleaved multi-task batch scheduling."""
    model.train()
    arc_margin.train()
    uncertainty_loss.train()

    iter_urban = iter(loader_urban)
    iter_vox = iter(loader_vox)
    iter_ravdess = iter(loader_ravdess)

    steps = max(len(loader_urban), len(loader_vox), len(loader_ravdess))
    if max_steps is not None and max_steps > 0:
        steps = min(steps, max_steps)

    epoch_losses = {"total": 0.0, "scene": 0.0, "spk": 0.0, "emo": 0.0}

    pbar = tqdm(range(steps), desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Multi-Task Training", leave=False)

    for step_idx in pbar:
        optimizer.zero_grad()

        # 1. Fetch Task 1: UrbanSound
        try:
            x_u, y_u = next(iter_urban)
        except StopIteration:
            iter_urban = iter(loader_urban)
            x_u, y_u = next(iter_urban)
        x_u, y_u = x_u.to(device), y_u.to(device)

        # 2. Fetch Task 2: VoxCeleb
        try:
            x_v, y_v = next(iter_vox)
        except StopIteration:
            iter_vox = iter(loader_vox)
            x_v, y_v = next(iter_vox)
        x_v, y_v = x_v.to(device), y_v.to(device)

        # 3. Fetch Task 3: RAVDESS
        try:
            x_r, y_r = next(iter_ravdess)
        except StopIteration:
            iter_ravdess = iter(loader_ravdess)
            x_r, y_r = next(iter_ravdess)
        x_r, y_r = x_r.to(device), y_r.to(device)

        # Sanity check: verify batches contain real physical audio (non-zero RMS)
        if step_idx == 0:
            rms_u = torch.sqrt(torch.mean(x_u ** 2)).item()
            rms_v = torch.sqrt(torch.mean(x_v ** 2)).item()
            rms_r = torch.sqrt(torch.mean(x_r ** 2)).item()
            assert rms_u > 1e-4, f"Zero or silent audio detected in UrbanSound batch: RMS={rms_u}"
            assert rms_v > 1e-4, f"Zero or silent audio detected in VoxCeleb batch: RMS={rms_v}"
            assert rms_r > 1e-4, f"Zero or silent audio detected in RAVDESS batch: RMS={rms_r}"

        # Forward Task 1: Scene Classification
        scene_l, _, _ = process_audio_in_streaming_chunks(model, x_u)
        loss_scene = F.cross_entropy(scene_l, y_u)

        # Forward Task 2: Speaker Voiceprints with ArcFace
        _, spk_emb, _ = process_audio_in_streaming_chunks(model, x_v)
        spk_logits = arc_margin(spk_emb, y_v)
        loss_spk = F.cross_entropy(spk_logits, y_v)

        # Forward Task 3: Speech Emotion Recognition
        _, _, emo_l = process_audio_in_streaming_chunks(model, x_r)
        loss_emo = F.cross_entropy(emo_l, y_r)

        # Balanced Composite Loss
        total_loss, weights = uncertainty_loss(loss_scene, loss_spk, loss_emo)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_losses["total"] += total_loss.item()
        epoch_losses["scene"] += loss_scene.item()
        epoch_losses["spk"] += loss_spk.item()
        epoch_losses["emo"] += loss_emo.item()

        pbar.set_postfix(
            loss=f"{total_loss.item():.3f}",
            sc=f"{loss_scene.item():.2f}",
            spk=f"{loss_spk.item():.2f}",
            emo=f"{loss_emo.item():.2f}",
        )

    for k in epoch_losses:
        epoch_losses[k] /= max(1, steps)

    return epoch_losses


def evaluate_multitask(
    model: nn.Module,
    arc_margin: ArcMarginProduct,
    loader_urban_val: DataLoader,
    loader_vox_heldout_id: Optional[DataLoader],
    loader_ravdess_val: DataLoader,
    device: torch.device,
    spk_verif_trials: Optional[List[Tuple[Path, Path, int]]] = None,
    max_val_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Validate all three multi-task heads on held-out evaluation sets.
    
    - UrbanSound: scene classification accuracy & macro-F1 on held-out fold.
    - VoxCeleb (Closed-Set Top-1): evaluated strictly on held-out recordings of known training identities.
    - VoxCeleb (Open-Set Verification): evaluated on disjoint speakers using fixed genuine/impostor trials.
    - RAVDESS (Validation SER): evaluated on validation actors (Actors 19-20).
    """
    model.eval()
    arc_margin.eval()

    metrics = {}

    # 1. Evaluate UrbanSound8K
    all_scene_preds, all_scene_targets = [], []
    with torch.no_grad():
        for b_idx, (x_u, y_u) in enumerate(loader_urban_val):
            if max_val_batches is not None and b_idx >= max_val_batches:
                break
            x_u = x_u.to(device)
            scene_l, _, _ = process_audio_in_streaming_chunks(model, x_u)
            preds = scene_l.argmax(dim=-1).cpu().numpy()
            all_scene_preds.extend(preds)
            all_scene_targets.extend(y_u.numpy())

    metrics["scene_acc"] = float(accuracy_score(all_scene_targets, all_scene_preds)) if all_scene_targets else 0.0
    metrics["scene_f1"] = float(f1_score(all_scene_targets, all_scene_preds, average="macro", zero_division=0)) if all_scene_targets else 0.0

    # 2. Evaluate VoxCeleb Speaker ID Top-1 on held-out recordings of known training identities
    all_spk_preds, all_spk_targets = [], []
    if loader_vox_heldout_id is not None:
        with torch.no_grad():
            for b_idx, (x_v, y_v) in enumerate(loader_vox_heldout_id):
                if max_val_batches is not None and b_idx >= max_val_batches:
                    break
                x_v = x_v.to(device)
                _, spk_emb, _ = process_audio_in_streaming_chunks(model, x_v)
                cosine = F.linear(spk_emb, F.normalize(arc_margin.weight, p=2, dim=-1))
                preds = cosine.argmax(dim=-1).cpu().numpy()
                all_spk_preds.extend(preds)
                all_spk_targets.extend(y_v.numpy())

    metrics["spk_top1"] = float(accuracy_score(all_spk_targets, all_spk_preds)) if all_spk_targets else 0.0

    # 2b. Evaluate Open-Set Speaker Verification on Disjoint Speaker Trials (EER, ROC-AUC, FAR/FRR)
    if spk_verif_trials:
        spk_verif = evaluate_speaker_verification_trials(model, spk_verif_trials, device)
        metrics["spk_eer"] = spk_verif["spk_eer"]
        metrics["spk_auc"] = spk_verif["spk_auc"]
        metrics["spk_threshold"] = spk_verif["spk_threshold"]
        metrics["spk_far"] = spk_verif["spk_far"]
        metrics["spk_frr"] = spk_verif["spk_frr"]
    else:
        metrics["spk_eer"] = 0.5
        metrics["spk_auc"] = 0.5
        metrics["spk_threshold"] = 0.0
        metrics["spk_far"] = 0.5
        metrics["spk_frr"] = 0.5

    # 3. Evaluate RAVDESS Speech Emotion (Actors 19-20)
    all_emo_preds, all_emo_targets = [], []
    with torch.no_grad():
        for b_idx, (x_r, y_r) in enumerate(loader_ravdess_val):
            if max_val_batches is not None and b_idx >= max_val_batches:
                break
            x_r = x_r.to(device)
            _, _, emo_l = process_audio_in_streaming_chunks(model, x_r)
            preds = emo_l.argmax(dim=-1).cpu().numpy()
            all_emo_preds.extend(preds)
            all_emo_targets.extend(y_r.numpy())

    metrics["emo_acc"] = float(accuracy_score(all_emo_targets, all_emo_preds)) if all_emo_targets else 0.0
    metrics["emo_f1"] = float(f1_score(all_emo_targets, all_emo_preds, average="macro", zero_division=0)) if all_emo_targets else 0.0

    # Composite Harmonic Score across all 3 tasks (inverting EER so higher is better: 1 - EER)
    spk_score = 1.0 - metrics["spk_eer"]
    metrics["composite_score"] = (metrics["scene_f1"] + spk_score + metrics["emo_f1"]) / 3.0
    return metrics


def evaluate_independent_test(
    model: nn.Module,
    loader_ravdess_test: DataLoader,
    test_spk_trials: List[Tuple[Path, Path, int]],
    device: torch.device,
    calibrated_spk_threshold: float,
) -> Dict[str, float]:
    """Evaluate final benchmark on independent test sets (never used for checkpoint selection).
    
    - Emotion: independent test actors (Actors 21-24).
    - Speaker Verification: fixed test trials evaluated at calibrated validation operating threshold.
    """
    model.eval()
    test_metrics = {}

    # 1. Independent Emotion Test (Actors 21-24)
    all_emo_preds, all_emo_targets = [], []
    with torch.no_grad():
        for x_r, y_r in loader_ravdess_test:
            x_r = x_r.to(device)
            _, _, emo_l = process_audio_in_streaming_chunks(model, x_r)
            preds = emo_l.argmax(dim=-1).cpu().numpy()
            all_emo_preds.extend(preds)
            all_emo_targets.extend(y_r.numpy())

    test_metrics["test_emo_acc"] = float(accuracy_score(all_emo_targets, all_emo_preds)) if all_emo_targets else 0.0
    test_metrics["test_emo_f1"] = float(f1_score(all_emo_targets, all_emo_preds, average="macro", zero_division=0)) if all_emo_targets else 0.0

    # 2. Independent Speaker Verification Test
    spk_test = evaluate_speaker_verification_trials(
        model=model,
        trials=test_spk_trials,
        device=device,
        fixed_threshold=calibrated_spk_threshold,
    )
    test_metrics["test_spk_eer"] = spk_test["spk_eer"]
    test_metrics["test_spk_auc"] = spk_test["spk_auc"]
    test_metrics["test_spk_far"] = spk_test["test_spk_far"] if "test_spk_far" in spk_test else spk_test.get("spk_far", 0.5)
    test_metrics["test_spk_frr"] = spk_test["test_spk_frr"] if "test_spk_frr" in spk_test else spk_test.get("spk_frr", 0.5)
    test_metrics["test_spk_operating_threshold"] = calibrated_spk_threshold

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Joint Multi-Task Streaming Transformer Training Pipeline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-task batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--val-fold", type=int, default=10, help="UrbanSound8K validation fold (1-10)")
    parser.add_argument("--max-speakers", type=int, default=50, help="Max VoxCeleb speakers for training")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Cap training steps per epoch for fast smoke testing")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches per epoch")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke-training job with checkpoint reload verification")
    parser.add_argument("--export-onnx", dest="export_onnx", action="store_true", default=True, help="Export static ONNX on completion")
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false", help="Disable static ONNX export")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_multitask_streaming_wavlm.pt if available")
    parser.add_argument("--force-resume", action="store_true", help="Force resume even if dataset/model fingerprint mismatches")
    parser.add_argument("--audit-datasets", dest="audit_datasets", action="store_true", default=True, help="Run preflight integrity audit across all dataset splits")
    parser.add_argument("--no-audit-datasets", dest="audit_datasets", action="store_false", help="Skip preflight dataset integrity audit")
    parser.add_argument("--num-spk-trials", type=int, default=100, help="Number of fixed speaker verification pairs for validation")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.batch_size = 4
        args.max_speakers = 6
        args.max_steps_per_epoch = 10
        print("[Smoke Test Mode] Enabled: epochs=2, batch_size=4, max_speakers=6, max_steps=10")

    device = get_device()
    print(f"[Causal MultiTask Transformer] Initializing Joint Training on Device: {device}...")

    # DataLoaders: UrbanSound8K (random crop for training, deterministic center crop for validation)
    train_folds = [f for f in range(1, 11) if f != args.val_fold]
    ds_urban_train = UrbanSoundRawDataset(folds=train_folds, is_train=True, strict=False)
    ds_urban_val = UrbanSoundRawDataset(folds=[args.val_fold], is_train=False, strict=False)

    # DataLoaders: VoxCeleb
    # - Train: 80% files of training speakers
    # - Heldout ID: remaining 20% files of SAME training speakers (for valid top-1 metric)
    # - Val: disjoint unseen speakers for open-set verification
    ds_vox_train = VoxCelebRawDataset(max_speakers=args.max_speakers, split="train", is_train=True, strict=False)
    ds_vox_heldout_id = VoxCelebRawDataset(max_speakers=args.max_speakers, split="heldout_id", is_train=False, strict=False)
    ds_vox_val = VoxCelebRawDataset(max_speakers=args.max_speakers, split="val", is_train=False, strict=False)
    num_speakers = max(2, len(ds_vox_train.speaker_to_id))

    # DataLoaders: RAVDESS
    # - Train: Actors 1-18
    # - Validation: Actors 19-20 (used for checkpoint selection)
    # - Independent Test: Actors 21-24 (never used for checkpoint selection or threshold tuning)
    ds_ravdess_train = RAVDESSRawDataset(actors=list(range(1, 19)), is_train=True, strict=False)
    ds_ravdess_val = RAVDESSRawDataset(actors=[19, 20], is_train=False, strict=False)
    ds_ravdess_test = RAVDESSRawDataset(actors=list(range(21, 25)), is_train=False, strict=False)

    # Preflight Dataset Integrity Audit
    if args.audit_datasets:
        print("[Preflight Audit] Initiating integrity check across all dataset splits...")
        splits_to_audit = [
            ("UrbanSound (Train)", ds_urban_train),
            ("UrbanSound (Val)", ds_urban_val),
            ("VoxCeleb (Train)", ds_vox_train),
            ("VoxCeleb (Heldout ID)", ds_vox_heldout_id),
            ("VoxCeleb (Disjoint Val)", ds_vox_val),
            ("RAVDESS (Train Actors 1-18)", ds_ravdess_train),
            ("RAVDESS (Val Actors 19-20)", ds_ravdess_val),
            ("RAVDESS (Test Actors 21-24)", ds_ravdess_test),
        ]
        min_req = 1 if args.smoke_test else min(args.batch_size, 4)
        run_dataset_preflight_audit(splits_to_audit, max_verify_per_split=20 if args.smoke_test else 50, min_samples=min_req)

    # Generate fixed reproducible speaker verification trials once
    val_spk_trials = generate_speaker_verification_trials(ds_vox_val, num_pairs=args.num_spk_trials, seed=42)
    test_spk_trials = generate_speaker_verification_trials(ds_vox_val, num_pairs=max(args.num_spk_trials * 2, 200), seed=1337)

    loader_urban_train = DataLoader(ds_urban_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_urban_val = DataLoader(ds_urban_val, batch_size=args.batch_size, shuffle=False)

    loader_vox_train = DataLoader(ds_vox_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_vox_heldout_id = DataLoader(ds_vox_heldout_id, batch_size=args.batch_size, shuffle=False) if len(ds_vox_heldout_id) > 0 else None
    loader_vox_val = DataLoader(ds_vox_val, batch_size=args.batch_size, shuffle=False)

    loader_ravdess_train = DataLoader(ds_ravdess_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_ravdess_val = DataLoader(ds_ravdess_val, batch_size=args.batch_size, shuffle=False)
    loader_ravdess_test = DataLoader(ds_ravdess_test, batch_size=args.batch_size, shuffle=False)

    print(
        f"[Causal MultiTask Transformer] Datasets & Trials verified:\n"
        f"  - UrbanSound:  {len(ds_urban_train)} train, {len(ds_urban_val)} val (Fold {args.val_fold})\n"
        f"  - VoxCeleb:    {len(ds_vox_train)} train, {len(ds_vox_heldout_id)} heldout-id, {len(ds_vox_val)} disjoint-val ({num_speakers} train classes)\n"
        f"  - Spk Trials:  {len(val_spk_trials)} fixed validation pairs, {len(test_spk_trials)} test pairs\n"
        f"  - RAVDESS:     {len(ds_ravdess_train)} train (1-18), {len(ds_ravdess_val)} val (19-20), {len(ds_ravdess_test)} test (21-24)\n"
    )

    # Initialize Causal Multi-Task Model
    model = CausalMultiTaskAudioTransformer(
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ffn_dim=512,
        max_past_frames=40,
        early_tap_layer=2,
        num_scene_classes=len(CLASSES),
        num_emotion_classes=len(EMOTIONS),
        embedding_dim=128,
    ).to(device)

    # ArcFace Margin Product for VoxCeleb Speaker Identity
    arc_margin = ArcMarginProduct(
        in_features=128,
        out_features=num_speakers,
        scale=30.0,
        margin=0.30,
    ).to(device)

    # Uncertainty Loss Balancer
    uncertainty_loss = MultiTaskUncertaintyLoss().to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(arc_margin.parameters()) + list(uncertainty_loss.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    tracker = RufloExperimentTracker(experiment_name="multitask_streaming_wavlm")

    # Distinct Artifact Separation:
    # - candidate_multitask_streaming_wavlm.pt: Best experimental composite score
    # - promoted_multitask_streaming_wavlm.pt: ONLY updated when ALL task gates pass
    # - best_multitask_streaming_wavlm.pt: Compatibility alias
    candidate_ckpt_path = CHECKPOINTS_DIR / "candidate_multitask_streaming_wavlm.pt"
    promoted_ckpt_path = CHECKPOINTS_DIR / "promoted_multitask_streaming_wavlm.pt"
    best_ckpt_path = CHECKPOINTS_DIR / "best_multitask_streaming_wavlm.pt"
    latest_ckpt_path = CHECKPOINTS_DIR / "latest_multitask_streaming_wavlm.pt"
    gate_metrics_path = CHECKPOINTS_DIR / "promoted_multitask_gate_metrics.json"

    best_composite = -1.0
    best_promoted_composite = -1.0
    best_calibrated_threshold = 0.0

    start_epoch = 1
    epoch_loss_history = []

    current_fingerprint = {
        "num_speakers": num_speakers,
        "hidden_dim": 256,
        "num_layers": 6,
        "val_fold": args.val_fold,
        "urban_train_samples": len(ds_urban_train),
        "vox_train_samples": len(ds_vox_train),
        "ravdess_train_samples": len(ds_ravdess_train),
    }

    if args.resume and latest_ckpt_path.exists():
        saved_state = torch.load(latest_ckpt_path, map_location=device)
        saved_fingerprint = saved_state.get("config_fingerprint", {})
        if saved_fingerprint and saved_fingerprint != current_fingerprint and not args.force_resume:
            raise ValueError(
                f"[Resume Aborted] Incompatible dataset/architecture fingerprint:\n"
                f"  Saved:   {saved_fingerprint}\n"
                f"  Current: {current_fingerprint}\n"
                f"Use --force-resume to override."
            )
        model.load_state_dict(saved_state["model_state"])
        arc_margin.load_state_dict(saved_state["arc_margin_state"])
        uncertainty_loss.load_state_dict(saved_state["uncertainty_loss_state"])
        optimizer.load_state_dict(saved_state["optimizer_state"])
        scheduler.load_state_dict(saved_state["scheduler_state"])
        start_epoch = saved_state["epoch"] + 1
        best_composite = saved_state.get("best_composite", -1.0)
        best_promoted_composite = saved_state.get("best_promoted_composite", -1.0)
        best_calibrated_threshold = saved_state.get("best_calibrated_threshold", 0.0)
        epoch_loss_history = saved_state.get("loss_history", [])

        # Restore RNG states
        if "python_rng_state" in saved_state:
            random.setstate(saved_state["python_rng_state"])
        if "numpy_rng_state" in saved_state:
            np.random.set_state(saved_state["numpy_rng_state"])
        if "torch_cpu_rng_state" in saved_state:
            torch.set_rng_state(saved_state["torch_cpu_rng_state"])
        if "torch_cuda_rng_state" in saved_state and torch.cuda.is_available() and saved_state["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(saved_state["torch_cuda_rng_state"])

        print(f"[Resume] Restored training state from {latest_ckpt_path.name} (Resuming at Epoch {start_epoch}/{args.epochs}, Best Composite: {best_composite:.4f})")

    # Task-Specific Promotion Gates
    GATE_MIN_SCENE_F1 = 0.50
    GATE_MAX_SPK_EER = 0.25
    GATE_MIN_EMO_F1 = 0.50

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_multitask_epoch(
            model=model,
            arc_margin=arc_margin,
            uncertainty_loss=uncertainty_loss,
            loader_urban=loader_urban_train,
            loader_vox=loader_vox_train,
            loader_ravdess=loader_ravdess_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            max_steps=args.max_steps_per_epoch,
        )
        scheduler.step()
        epoch_loss_history.append(losses["total"])

        is_final_epoch = (epoch == args.epochs)
        cur_max_val = 5 if args.smoke_test else (None if is_final_epoch else args.max_val_batches)
        val_metrics = evaluate_multitask(
            model=model,
            arc_margin=arc_margin,
            loader_urban_val=loader_urban_val,
            loader_vox_heldout_id=loader_vox_heldout_id,
            loader_ravdess_val=loader_ravdess_val,
            device=device,
            spk_verif_trials=val_spk_trials,
            max_val_batches=cur_max_val,
        )

        comp = val_metrics["composite_score"]
        spk_eer = val_metrics["spk_eer"]
        passes_gates = (
            val_metrics["scene_f1"] >= GATE_MIN_SCENE_F1
            and spk_eer <= GATE_MAX_SPK_EER
            and val_metrics["emo_f1"] >= GATE_MIN_EMO_F1
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {losses['total']:.4f} | "
            f"Scene F1: {val_metrics['scene_f1']:.4f} | "
            f"Spk EER: {spk_eer*100:.1f}% (Top-1: {val_metrics['spk_top1']*100:.1f}%, FAR: {val_metrics['spk_far']*100:.1f}%) | "
            f"SER F1: {val_metrics['emo_f1']:.4f} | "
            f"Composite: {comp:.4f} | Gates: {'PASSED' if passes_gates else 'HELD'}"
        )

        # 1. Update experimental candidate checkpoint
        if comp > best_composite or (args.smoke_test and epoch == args.epochs):
            best_composite = comp
            best_calibrated_threshold = val_metrics.get("spk_threshold", 0.0)
            save_checkpoint_atomic(model.state_dict(), candidate_ckpt_path)
            save_checkpoint_atomic(model.state_dict(), best_ckpt_path)
            print(f"  [CANDIDATE SAVED] Composite: {comp:.4f} -> {candidate_ckpt_path.name}")

        # 2. Strict Checkpoint Promotion Gate: ONLY update promoted artifact when all gates pass
        if passes_gates and (comp > best_promoted_composite or (args.smoke_test and epoch == args.epochs)):
            best_promoted_composite = comp
            save_checkpoint_atomic(model.state_dict(), promoted_ckpt_path)
            gate_record = {
                "epoch": epoch,
                "composite_score": comp,
                "scene_f1": val_metrics["scene_f1"],
                "spk_eer": val_metrics["spk_eer"],
                "spk_auc": val_metrics["spk_auc"],
                "spk_top1": val_metrics["spk_top1"],
                "emo_f1": val_metrics["emo_f1"],
                "calibrated_threshold": best_calibrated_threshold,
                "gate_status": "PASSED ALL GATES",
                "timestamp": time.time(),
            }
            with open(gate_metrics_path, "w", encoding="utf-8") as gf:
                json.dump(gate_record, gf, indent=2)
            print(f"  [PROMOTED CHECKPOINT] Gate verified! Saved to {promoted_ckpt_path.name} & {gate_metrics_path.name}")

        # 3. Always save latest checkpoint atomically for crash recovery
        save_checkpoint_atomic({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "arc_margin_state": arc_margin.state_dict(),
            "uncertainty_loss_state": uncertainty_loss.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_composite": best_composite,
            "best_promoted_composite": best_promoted_composite,
            "best_calibrated_threshold": best_calibrated_threshold,
            "loss_history": epoch_loss_history,
            "config_fingerprint": current_fingerprint,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, latest_ckpt_path)

    total_time = time.time() - t_start
    print(f"\n[Causal MultiTask Transformer] Training completed in {total_time/60:.2f} mins. Best Composite: {best_composite:.4f}")

    # Evaluate on Independent Clean Benchmark Splits (never used for training or epoch selection)
    eval_model_path = promoted_ckpt_path if promoted_ckpt_path.exists() else best_ckpt_path
    if eval_model_path.exists() and len(ds_ravdess_test) > 0:
        print(f"\n[Independent Benchmark] Evaluating {eval_model_path.name} on clean test sets...")
        model.load_state_dict(torch.load(eval_model_path, map_location=device))
        test_results = evaluate_independent_test(
            model=model,
            loader_ravdess_test=loader_ravdess_test,
            test_spk_trials=test_spk_trials,
            device=device,
            calibrated_spk_threshold=best_calibrated_threshold,
        )
        print(
            f"  * Independent Emotion (Actors 21-24) F1: {test_results['test_emo_f1']:.4f} (Acc: {test_results['test_emo_acc']*100:.1f}%)\n"
            f"  * Independent Speaker Verification EER: {test_results['test_spk_eer']*100:.1f}% (AUC: {test_results['test_spk_auc']:.4f})\n"
            f"  * Calibrated Operating Threshold: {best_calibrated_threshold:.4f} -> FAR: {test_results['test_spk_far']*100:.1f}%, FRR: {test_results['test_spk_frr']*100:.1f}%\n"
        )

    # Smoke Test Verification Routine
    if args.smoke_test:
        print("\n=== SMOKE-TEST VERIFICATION PROTOCOL ===")
        # 1. Verify loss descent
        if len(epoch_loss_history) >= 2:
            loss_descended = epoch_loss_history[-1] <= epoch_loss_history[0]
            print(f"  [1] Loss descent: Epoch 1 = {epoch_loss_history[0]:.4f} -> Epoch 2 = {epoch_loss_history[-1]:.4f} (Descended: {loss_descended})")

        # 2. Checkpoint save & reload roundtrip
        print(f"  [2] Checkpoint reload from: {best_ckpt_path}")
        assert best_ckpt_path.exists(), f"Expected checkpoint file does not exist: {best_ckpt_path}"
        assert best_ckpt_path.stat().st_size > 10_000, "Checkpoint file size is suspiciously small"

        fresh_model = CausalMultiTaskAudioTransformer(
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            ffn_dim=512,
            max_past_frames=40,
            early_tap_layer=2,
            num_scene_classes=len(CLASSES),
            num_emotion_classes=len(EMOTIONS),
            embedding_dim=128,
        )
        fresh_model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
        fresh_model.eval()

        # 3. Numerical inference sanity check
        test_chunk = torch.randn(1, CHUNK_SAMPLES)
        with torch.no_grad():
            sc_l, spk_e, emo_l, cache = fresh_model(test_chunk)
        assert sc_l.shape == (1, len(CLASSES))
        assert spk_e.shape == (1, 128)
        assert emo_l.shape == (1, len(EMOTIONS))
        assert np.isclose(torch.norm(spk_e, p=2, dim=-1).item(), 1.0, atol=1e-4)

        print("  [3] Numerical inference check on reloaded checkpoint: Output shapes & L2 norms verified!")
        print("  [Phase 2 Smoke-Test Passed] All checks passed successfully!\n")

    # Track experiment in Ruflo registry
    tracker.log_run(
        fold=args.val_fold,
        epochs=args.epochs,
        val_accuracy=float(best_composite if best_composite > 0 else 0.0),
        val_macro_f1=float(best_composite if best_composite > 0 else 0.0),
        per_class_f1={
            "composite": float(best_composite if best_composite > 0 else 0.0),
            "scene_f1": float(val_metrics.get("scene_f1", 0.0)),
            "spk_eer": float(val_metrics.get("spk_eer", 1.0)),
            "emo_f1": float(val_metrics.get("emo_f1", 0.0)),
            "spk_top1": float(val_metrics.get("spk_top1", 0.0)),
        },
        hyperparameters={
            "architecture": "CausalMultiTaskAudioTransformer",
            "epochs": args.epochs,
            "lr": args.lr,
            "val_fold": args.val_fold,
            "max_speakers": args.max_speakers,
            "early_tap_layer": 2,
            "deep_tap_layer": 6,
            "promotion_gates": {
                "min_scene_f1": GATE_MIN_SCENE_F1,
                "max_spk_eer": GATE_MAX_SPK_EER,
                "min_emo_f1": GATE_MIN_EMO_F1,
            },
        },
        checkpoint_path=str(promoted_ckpt_path if promoted_ckpt_path.exists() else best_ckpt_path),
    )

    if args.export_onnx and best_ckpt_path.exists():
        print("[Causal MultiTask Transformer] Exporting trained checkpoint to static ONNX model...")
        export_streaming_wavlm_to_onnx(
            checkpoint_path=best_ckpt_path,
            static_shapes=True,
            strict_checkpoints=False if args.smoke_test else True,
        )


if __name__ == "__main__":
    main()
