"""Unified Speech & Audio Intelligence Pipeline.
Multi-task audio processing combining UrbanSound8K, VoxCeleb, and RAVDESS.
"""

__version__ = "1.0.0"
__all__ = [
    "audio_loader",
    "config",
    "dataset",
    "device",
    "diarization",
    "emotion_dataset",
    "emotion_model",
    "inference",
    "model",
    "ruflo_tracker",
    "speaker_id",
    "streaming_wavlm",
    "train",
    "train_emotion",
    "train_speaker_id",
    "train_streaming_wavlm",
    "unified_analyzer",
    "urban_noise_mixer",
    "validate_environment",
]
