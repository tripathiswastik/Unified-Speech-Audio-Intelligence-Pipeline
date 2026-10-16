"""Unified Audio Intelligence Engine.
Performs:
  1. Environmental Acoustic Scene Detection (UrbanSound8K)
  2. Speaker Diarization ('Who spoke when?')
  3. Speaker Identification (VoxCeleb Voiceprints)
  4. Speech Emotion Recognition (RAVDESS SER)
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import soundfile as sf

from pipeline.config import BASE_DIR, CHECKPOINTS_DIR, CLASSES, ID_TO_CLASS, SAMPLE_RATE
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.model import AudioCNN
from pipeline.emotion_model import EmotionClassifierCNN
from pipeline.emotion_dataset import EMOTIONS
from pipeline.speaker_id import SpeakerEmbeddingNet, SpeakerIdentifier
from pipeline.diarization import SpeakerDiarizer
from pipeline.device import get_device
from pipeline.ruflo_tracker import RufloExperimentTracker


class UnifiedAudioAnalyzer:
    """Multi-task audio processing pipeline combining UrbanSound, Speaker ID, Emotion & Diarization."""

    def __init__(self, device: Optional[torch.device] = None, strict_checkpoints: bool = False):
        self.device = device if device is not None else get_device()
        self.strict_checkpoints = strict_checkpoints

        # 1. Environmental Audio Classifier (UrbanSound8K)
        self.urban_model = AudioCNN().to(self.device)
        self.urban_checkpoint_loaded = False
        chk = sorted(list(CHECKPOINTS_DIR.glob("best_model_fold*.pt")))
        if chk:
            state = torch.load(chk[0], map_location=self.device)
            self.urban_model.load_state_dict(state["model_state"])
            self.urban_checkpoint_loaded = True
        elif self.strict_checkpoints:
            raise FileNotFoundError(f"UrbanSound checkpoint missing in {CHECKPOINTS_DIR}. Train or download fold model.")
        self.urban_model.eval()

        # 2. Emotion Classifier (RAVDESS)
        self.emotion_model = EmotionClassifierCNN().to(self.device)
        self.emotion_checkpoint_loaded = False
        emo_chk = CHECKPOINTS_DIR / "best_emotion_model.pt"
        if emo_chk.exists():
            emo_state = torch.load(emo_chk, map_location=self.device)
            self.emotion_model.load_state_dict(emo_state["model_state"])
            self.emotion_checkpoint_loaded = True
        elif self.strict_checkpoints:
            raise FileNotFoundError(f"Emotion checkpoint best_emotion_model.pt missing in {CHECKPOINTS_DIR}.")
        self.emotion_model.eval()

        # 3. Speaker ID & Diarizer
        self.speaker_net = SpeakerEmbeddingNet().to(self.device)
        self.speaker_checkpoint_loaded = False
        spk_chk = CHECKPOINTS_DIR / "best_speaker_model.pt"
        if spk_chk.exists():
            spk_state = torch.load(spk_chk, map_location=self.device)
            self.speaker_net.load_state_dict(spk_state["model_state"])
            self.speaker_checkpoint_loaded = True
        elif self.strict_checkpoints:
            raise FileNotFoundError(f"Speaker embedding checkpoint best_speaker_model.pt missing in {CHECKPOINTS_DIR}.")
        self.speaker_net.eval()

        self.speaker_identifier = SpeakerIdentifier(model=self.speaker_net, device=self.device)
        self.diarizer = SpeakerDiarizer(embedding_net=self.speaker_net, device=self.device)

        # 4. Ruflo Tracker
        self.tracker = RufloExperimentTracker(experiment_name="unified_speech_intelligence")

    def enroll_known_speaker(self, name: str, audio_path: Path):
        """Register a known speaker's voiceprint from a reference audio file."""
        self.speaker_identifier.enroll(name, audio_path)

    def analyze_audio_file(
        self, audio_path: Path, num_speakers: int = 2
    ) -> Dict[str, Any]:
        """Perform end-to-end multi-task analysis on an audio recording."""
        audio_path = Path(audio_path)
        waveform = load_and_preprocess_audio(audio_path)

        # Step 1: Detect Environmental Acoustic Background (UrbanSound8K)
        spec = compute_log_mel_spectrogram(waveform)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            urban_logits = self.urban_model(spec_tensor)
            urban_probs = torch.softmax(urban_logits, dim=-1)[0].cpu().numpy()
            top_urban_idx = int(np.argmax(urban_probs))
            env_scene = ID_TO_CLASS[top_urban_idx]
            env_conf = float(urban_probs[top_urban_idx])

        # Step 2: Speaker Diarization
        segments = self.diarizer.diarize(waveform, num_speakers=num_speakers)
        turns_analysis = []

        # Step 3 & 4: Speaker ID & Emotion Recognition per Turn
        for seg in segments:
            # Segment waveform
            st_idx = int(seg.start * SAMPLE_RATE)
            et_idx = int(seg.end * SAMPLE_RATE)
            seg_wave = waveform[st_idx:et_idx]

            # Speaker identification
            identified_spk, spk_conf = self.speaker_identifier.identify(seg_wave)

            # Emotion prediction
            seg_spec = compute_log_mel_spectrogram(seg_wave)
            seg_tensor = torch.from_numpy(seg_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                emo_logits = self.emotion_model(seg_tensor)
                emo_probs = torch.softmax(emo_logits, dim=-1)[0].cpu().numpy()
                top_emo_idx = int(np.argmax(emo_probs))
                emotion_label = EMOTIONS[top_emo_idx]
                emotion_conf = float(emo_probs[top_emo_idx])

            turns_analysis.append({
                "time_range": f"{seg.start:.2f}s - {seg.end:.2f}s",
                "cluster_speaker": seg.speaker,
                "identified_speaker": identified_spk,
                "speaker_confidence": round(spk_conf, 3),
                "emotion": emotion_label,
                "emotion_confidence": round(emotion_conf, 3),
            })

        result = {
            "file_name": audio_path.name,
            "duration": f"{len(waveform)/SAMPLE_RATE:.2f}s",
            "model_provenance": {
                "environmental_classifier_trained": self.urban_checkpoint_loaded,
                "emotion_classifier_trained": self.emotion_checkpoint_loaded,
                "speaker_embedding_trained": self.speaker_checkpoint_loaded,
            },
            "environmental_acoustic_scene": {
                "detected_class": env_scene,
                "confidence": round(env_conf, 3),
                "is_trained_model": self.urban_checkpoint_loaded,
            },
            "total_speech_turns": len(turns_analysis),
            "speaker_diarization_and_emotions": turns_analysis,
        }
        return result


if __name__ == "__main__":
    import json
    analyzer = UnifiedAudioAnalyzer()

    # Enroll two sample speakers from VoxCeleb if available
    vox_root = BASE_DIR / "wav"
    spk1_dir = vox_root / "id10001"
    spk2_dir = vox_root / "id10002"

    if spk1_dir.exists():
        wav1 = list(spk1_dir.glob("*/*.wav"))
        if wav1:
            analyzer.enroll_known_speaker("Speaker_id10001", wav1[0])
    if spk2_dir.exists():
        wav2 = list(spk2_dir.glob("*/*.wav"))
        if wav2:
            analyzer.enroll_known_speaker("Speaker_id10002", wav2[0])

    # Test on a RAVDESS sample
    sample_speech = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    if sample_speech.exists():
        print(f"\nAnalyzing sample audio: {sample_speech.name}...")
        report = analyzer.analyze_audio_file(sample_speech, num_speakers=1)
        print("\n--- Unified Analysis Report ---")
        print(json.dumps(report, indent=2))
    else:
        print(f"Sample audio {sample_speech} not found. Skipping inference demo.")
