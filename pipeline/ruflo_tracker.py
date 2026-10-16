"""Ruflo Integration & Experiment Lineage Tracker for UrbanSound.
Records training metadata, cross-validation metrics, and agent topology logs.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import time
import numpy as np

from pipeline.config import OUTPUT_DIR, CLASSES


class RufloExperimentTracker:
    def __init__(self, experiment_name: str = "urbansound_cnn_specaugment"):
        self.experiment_name = experiment_name
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "ruflo_experiment_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "project": "UrbanSound8K",
            "taxonomy": CLASSES,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": [],
        }

    def log_run(
        self,
        fold: int,
        epochs: int,
        val_accuracy: float,
        val_macro_f1: float,
        per_class_f1: Dict[str, float],
        hyperparameters: Dict[str, Any],
        checkpoint_path: str,
    ):
        """Log a fold training run into Ruflo memory/lineage store."""
        run_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiment": self.experiment_name,
            "test_fold": fold,
            "epochs": epochs,
            "val_accuracy": round(float(val_accuracy), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "per_class_f1": {k: round(float(v), 4) for k, v in per_class_f1.items()},
            "hyperparameters": hyperparameters,
            "checkpoint": checkpoint_path,
        }
        temp_file = self.history_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        temp_file.replace(self.history_file)
        print(f"[Ruflo Tracker] Logged fold {fold} metrics -> Macro F1: {val_macro_f1:.4f}, Accuracy: {val_accuracy:.4f}")

    def get_cv_summary(self) -> Dict[str, Any]:
        """Compute aggregate cross-validation statistics across all logged fold runs."""
        self.history = self._load_history()
        runs = self.history.get("runs", [])
        if not runs:
            return {"total_folds": 0}
        accs = [r["val_accuracy"] for r in runs]
        f1s = [r["val_macro_f1"] for r in runs]
        return {
            "total_folds": len(runs),
            "mean_val_accuracy": round(sum(accs) / len(accs), 4),
            "mean_val_macro_f1": round(sum(f1s) / len(f1s), 4),
            "std_val_macro_f1": round(float(np.std(f1s)), 4) if len(f1s) > 1 else 0.0,
        }
