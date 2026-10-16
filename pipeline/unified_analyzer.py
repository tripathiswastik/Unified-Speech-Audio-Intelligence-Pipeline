"""Unified Audio Intelligence Engine.
Performs:
  1. Continuous Environmental Acoustic Scene Detection (UrbanSound8K)
  2. Full-Length Speaker Diarization ('Who spoke when?')
  3. Speaker Identification (VoxCeleb Voiceprints)
  4. Speech Emotion Recognition (RAVDESS SER)

Enforces fail-closed checkpoint validation by default, allowing random weights
only when demo_mode=True is explicitly requested.
Supports direct zero-disk in-memory waveform analysis.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import numpy as np
import torch
import soundfile as sf

from pipeline.config import BASE_DIR, CHECKPOINTS_DIR, CLASSES, ID_TO_CLASS, SAMPLE_RATE, TARGET_SAMPLES
from pipeline.audio_loader import (
    load_full_audio,
    preprocess_audio_window,
    compute_log_mel_spectrogram,
    compute_cmvn_log_mel_spectrogram,
)
from pipeline.model import AudioCNN, SEAudioCNN, SpeakerEmbeddingNet, SESpeakerEmbeddingNet
from pipeline.emotion_model import EmotionClassifierCNN, SEEmotionClassifierCNN, SSLEmotionClassifier
from pipeline.emotion_dataset import EMOTIONS
from pipeline.speaker_id import SpeakerIdentifier
from pipeline.diarization import SpeakerDiarizer
from pipeline.device import get_device
from pipeline.ruflo_tracker import RufloExperimentTracker



def compute_file_sha256(path: Path) -> str:
    """Calculate truncated SHA256 checksum for checkpoint verification."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "unknown"


class UnifiedAudioAnalyzer:
    """Multi-task audio processing pipeline combining UrbanSound, Speaker ID, Emotion & Diarization."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        strict_checkpoints: bool = True,
        demo_mode: bool = False,
    ):
        self.device = device if device is not None else get_device()
        self.strict_checkpoints = strict_checkpoints
        self.demo_mode = demo_mode

        # 1. Environmental Audio Classifier (UrbanSound8K)
        self.urban_checkpoint_loaded = False
        self.urban_checkpoint_path: Optional[Path] = None
        self.urban_checkpoint_sha256: Optional[str] = None
        self.use_cmvn = False

        chk = sorted(list(CHECKPOINTS_DIR.glob("best_model_fold*.pt")))
        if chk:
            self.urban_checkpoint_path = chk[0]
            state = torch.load(chk[0], map_location=self.device)
            model_type = state.get("model_type", "baseline")
            self.use_cmvn = state.get("use_cmvn", False)
            if model_type == "se_cnn":
                self.urban_model = SEAudioCNN().to(self.device)
            else:
                self.urban_model = AudioCNN().to(self.device)
            self.urban_model.load_state_dict(state["model_state"])
            self.urban_checkpoint_loaded = True
            self.urban_checkpoint_sha256 = compute_file_sha256(chk[0])
        elif self.strict_checkpoints and not self.demo_mode:
            raise FileNotFoundError(
                f"Required UrbanSound checkpoint missing in {CHECKPOINTS_DIR}. "
                "Train fold model or pass demo_mode=True to run with synthetic demo weights."
            )
        else:
            self.urban_model = AudioCNN().to(self.device)
        self.urban_model.eval()

        # 2. Emotion Classifier (RAVDESS)
        self.emotion_checkpoint_loaded = False
        self.emotion_checkpoint_path: Optional[Path] = None
        self.emotion_checkpoint_sha256: Optional[str] = None

        emo_chk = CHECKPOINTS_DIR / "best_emotion_model.pt"
        if emo_chk.exists():
            self.emotion_checkpoint_path = emo_chk
            emo_state = torch.load(emo_chk, map_location=self.device)
            emo_model_type = emo_state.get("model_type", "baseline")
            if emo_model_type == "se_cnn":
                self.emotion_model = SEEmotionClassifierCNN().to(self.device)
            elif emo_model_type == "ssl":
                self.emotion_model = SSLEmotionClassifier().to(self.device)
            else:
                self.emotion_model = EmotionClassifierCNN().to(self.device)
            self.emotion_model.load_state_dict(emo_state["model_state"])
            self.emotion_checkpoint_loaded = True
            self.emotion_checkpoint_sha256 = compute_file_sha256(emo_chk)
        elif self.strict_checkpoints and not self.demo_mode:
            raise FileNotFoundError(
                f"Required Emotion checkpoint best_emotion_model.pt missing in {CHECKPOINTS_DIR}. "
                "Train emotion model or pass demo_mode=True to run with synthetic demo weights."
            )
        else:
            self.emotion_model = EmotionClassifierCNN().to(self.device)
        self.emotion_model.eval()

        # 3. Speaker ID & Diarizer (VoxCeleb)
        self.speaker_checkpoint_loaded = False
        self.speaker_checkpoint_path: Optional[Path] = None
        self.speaker_checkpoint_sha256: Optional[str] = None

        spk_chk = CHECKPOINTS_DIR / "best_speaker_model.pt"
        if spk_chk.exists():
            self.speaker_checkpoint_path = spk_chk
            spk_state = torch.load(spk_chk, map_location=self.device)
            spk_backbone = spk_state.get("backbone", "baseline")
            self.speaker_net = SpeakerEmbeddingNet(backbone=spk_backbone).to(self.device)
            self.speaker_net.load_state_dict(spk_state["model_state"])
            self.speaker_checkpoint_loaded = True
            self.speaker_checkpoint_sha256 = compute_file_sha256(spk_chk)
        elif self.strict_checkpoints and not self.demo_mode:
            raise FileNotFoundError(
                f"Required Speaker embedding checkpoint best_speaker_model.pt missing in {CHECKPOINTS_DIR}. "
                "Train speaker model or pass demo_mode=True to run with synthetic demo weights."
            )
        else:
            self.speaker_net = SpeakerEmbeddingNet().to(self.device)
        self.speaker_net.eval()


        self.speaker_identifier = SpeakerIdentifier(model=self.speaker_net, device=self.device)
        self.diarizer = SpeakerDiarizer(embedding_net=self.speaker_net, device=self.device)

        # 4. Ruflo Tracker
        self.tracker = RufloExperimentTracker(experiment_name="unified_speech_intelligence")

    def enroll_known_speaker(self, name: str, audio_path: Path):
        """Register a known speaker's voiceprint from a reference audio file."""
        self.speaker_identifier.enroll(name, audio_path)

    def analyze_waveform(
        self,
        waveform: np.ndarray,
        file_name: str = "in_memory_audio",
        num_speakers: Optional[int] = 2,
    ) -> Dict[str, Any]:
        """Perform end-to-end multi-task analysis directly on an in-memory waveform."""
        true_duration = float(len(waveform)) / float(SAMPLE_RATE) if len(waveform) > 0 else 0.0

        if len(waveform) == 0:
            return {
                "file_name": file_name,
                "duration": "0.00s",
                "error": "Empty or unreadable audio waveform.",
                "total_speech_turns": 0,
                "speaker_diarization_and_emotions": [],
            }

        # Step 1: Detect Environmental Acoustic Background across Continuous Audio
        # Vectorized batch processing across 4.0s windows
        spec_fn = compute_cmvn_log_mel_spectrogram if getattr(self, "use_cmvn", False) else compute_log_mel_spectrogram
        window_samples = TARGET_SAMPLES
        chunks = []
        if len(waveform) <= window_samples:
            chunks.append(preprocess_audio_window(waveform, target_samples=window_samples))
        else:
            for st in range(0, len(waveform) - window_samples + 1, window_samples):
                chunks.append(waveform[st : st + window_samples])

            # Evaluate partial tail window to guarantee full coverage
            tail_samples = len(waveform) % window_samples
            if tail_samples >= int(0.25 * SAMPLE_RATE):
                chunks.append(waveform[-window_samples:])

            if not chunks:
                chunks.append(preprocess_audio_window(waveform, target_samples=window_samples))

        specs_list = [spec_fn(c) for c in chunks]
        batch_spec_tensor = torch.from_numpy(np.stack(specs_list, axis=0)).unsqueeze(1).float().to(self.device)
        with torch.no_grad():
            batch_logits = self.urban_model(batch_spec_tensor)
            batch_probs = torch.softmax(batch_logits, dim=-1).cpu().numpy()
            urban_probs = np.mean(batch_probs, axis=0)

        top_urban_idx = int(np.argmax(urban_probs))
        env_scene = ID_TO_CLASS[top_urban_idx]
        env_conf = float(urban_probs[top_urban_idx])

        # Step 2: Full-Recording Speaker Diarization with timeline invariants
        segments = self.diarizer.diarize(waveform, num_speakers=num_speakers)

        # Step 3 & 4: Batched Speaker ID & Emotion Recognition per Speech Turn
        valid_segments = []
        seg_waves = []
        turn_embeddings = []

        for seg in segments:
            st_idx = max(0, int(seg.start * SAMPLE_RATE))
            et_idx = min(len(waveform), int(seg.end * SAMPLE_RATE))
            if et_idx > st_idx:
                valid_segments.append(seg)
                seg_waves.append(waveform[st_idx:et_idx])
                turn_embeddings.append(seg.embedding)

        turns_analysis = []
        if valid_segments:
            # 1. Batched Speaker Identification reusing precomputed turn embeddings from diarization
            turn_embs_arr = np.stack(turn_embeddings, axis=0)
            id_results = self.speaker_identifier.identify_batch(turn_embs_arr)

            # 2. Batched Emotion Recognition across all speech turns in a single vectorized forward pass
            padded_turns = [preprocess_audio_window(w, target_samples=TARGET_SAMPLES) for w in seg_waves]
            turn_specs = [compute_log_mel_spectrogram(w) for w in padded_turns]
            batch_emo_tensor = torch.from_numpy(np.stack(turn_specs, axis=0)).unsqueeze(1).float().to(self.device)

            with torch.no_grad():
                emo_logits = self.emotion_model(batch_emo_tensor)
                emo_probs = torch.softmax(emo_logits, dim=-1).cpu().numpy()

            for i, seg in enumerate(valid_segments):
                identified_spk, spk_conf = id_results[i]
                probs_i = emo_probs[i]
                top_emo_idx = int(np.argmax(probs_i))
                turns_analysis.append({
                    "time_range": f"{seg.start:.2f}s - {seg.end:.2f}s",
                    "cluster_speaker": seg.speaker,
                    "identified_speaker": identified_spk,
                    "speaker_confidence": round(spk_conf, 3),
                    "emotion": EMOTIONS[top_emo_idx],
                    "emotion_confidence": round(float(probs_i[top_emo_idx]), 3),
                })

        result = {

            "file_name": file_name,
            "duration": f"{true_duration:.2f}s",
            "model_provenance": {
                "is_synthetic_demo_mode": self.demo_mode,
                "environmental_classifier": {
                    "is_trained": self.urban_checkpoint_loaded,
                    "checkpoint": self.urban_checkpoint_path.name if self.urban_checkpoint_path else "None (Untrained)",
                    "sha256": self.urban_checkpoint_sha256,
                },
                "emotion_classifier": {
                    "is_trained": self.emotion_checkpoint_loaded,
                    "checkpoint": self.emotion_checkpoint_path.name if self.emotion_checkpoint_path else "None (Untrained)",
                    "sha256": self.emotion_checkpoint_sha256,
                },
                "speaker_embedding": {
                    "is_trained": self.speaker_checkpoint_loaded,
                    "checkpoint": self.speaker_checkpoint_path.name if self.speaker_checkpoint_path else "None (Untrained)",
                    "sha256": self.speaker_checkpoint_sha256,
                },
            },
            "environmental_acoustic_scene": {
                "detected_class": env_scene,
                "confidence": round(env_conf, 3),
                "is_trained_model": self.urban_checkpoint_loaded,
                "disclaimer": "Predicted environmental acoustic context; represents acoustic classification, not absolute physical confirmation.",
            },
            "total_speech_turns": len(turns_analysis),
            "speaker_diarization_and_emotions": turns_analysis,
        }
        return result

    def analyze_audio_file(
        self, audio_path: Path, num_speakers: Optional[int] = 2
    ) -> Dict[str, Any]:
        """Perform end-to-end multi-task analysis on an audio file on disk."""
        audio_path = Path(audio_path)
        waveform, _ = load_full_audio(audio_path)
        return self.analyze_waveform(waveform, file_name=audio_path.name, num_speakers=num_speakers)


if __name__ == "__main__":
    import json
    analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=True)

    # Enroll sample speaker if available
    vox_root = BASE_DIR / "wav"
    spk1_dir = vox_root / "id10001"
    if spk1_dir.exists():
        wav1 = list(spk1_dir.glob("*/*.wav"))
        if wav1:
            analyzer.enroll_known_speaker("Speaker_id10001", wav1[0])

    # Test on a RAVDESS sample
    sample_speech = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    if sample_speech.exists():
        print(f"\nAnalyzing sample audio: {sample_speech.name}...")
        report = analyzer.analyze_audio_file(sample_speech, num_speakers=1)
        print("\n--- Unified Analysis Report ---")
        print(json.dumps(report, indent=2))
    else:
        print(f"Sample audio {sample_speech} not found. Skipping inference demo.")
