"""Tests for UnifiedAudioAnalyzer multi-task inference pipeline."""
import numpy as np
import soundfile as sf
import pytest
from pipeline.config import SAMPLE_RATE, BASE_DIR
from pipeline.unified_analyzer import UnifiedAudioAnalyzer


def test_unified_analyzer_provenance_and_inference(tmp_path):
    """Verify end-to-end multi-task analysis produces expected contract on synthetic audio."""
    analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False)
    assert hasattr(analyzer, "urban_checkpoint_loaded")
    assert hasattr(analyzer, "emotion_checkpoint_loaded")
    assert hasattr(analyzer, "speaker_checkpoint_loaded")

    # Generate synthetic 4-second WAV fixture
    duration = 4.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    synth_wave = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    synth_file = tmp_path / "synthetic_test.wav"
    sf.write(str(synth_file), synth_wave, SAMPLE_RATE)

    report = analyzer.analyze_audio_file(synth_file, num_speakers=2)

    # Validate schema contract
    assert "file_name" in report
    assert "duration" in report
    assert "model_provenance" in report
    assert "environmental_acoustic_scene" in report
    assert "speaker_diarization_and_emotions" in report

    scene = report["environmental_acoustic_scene"]
    assert "detected_class" in scene
    assert "confidence" in scene
    assert "is_trained_model" in scene
    assert 0.0 <= scene["confidence"] <= 1.0


def test_strict_checkpoint_failure_on_nonexistent(monkeypatch, tmp_path):
    """Verify strict_checkpoints raises FileNotFoundError when checkpoints are missing."""
    import pipeline.unified_analyzer as ua
    empty_dir = tmp_path / "empty_checkpoints"
    empty_dir.mkdir()
    monkeypatch.setattr(ua, "CHECKPOINTS_DIR", empty_dir)

    with pytest.raises(FileNotFoundError):
        ua.UnifiedAudioAnalyzer(strict_checkpoints=True, demo_mode=False)


def test_demo_mode_allows_missing_checkpoints(monkeypatch, tmp_path):
    """Verify demo_mode=True allows instantiation even when checkpoints are missing."""
    import pipeline.unified_analyzer as ua
    empty_dir = tmp_path / "empty_checkpoints_demo"
    empty_dir.mkdir()
    monkeypatch.setattr(ua, "CHECKPOINTS_DIR", empty_dir)

    analyzer = ua.UnifiedAudioAnalyzer(strict_checkpoints=True, demo_mode=True)
    assert analyzer.demo_mode is True
    assert analyzer.urban_checkpoint_loaded is False


def test_unified_analyzer_analyze_waveform_in_memory():
    """Verify analyze_waveform runs directly on in-memory array with zero disk I/O."""
    analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=True)
    # 7 seconds of audio (multi-window with partial tail)
    t = np.linspace(0, 7.0, int(SAMPLE_RATE * 7.0), endpoint=False)
    synth_wave = (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    report = analyzer.analyze_waveform(synth_wave, file_name="in_memory_stream.wav", num_speakers=1)
    assert report["file_name"] == "in_memory_stream.wav"
    assert report["duration"] == "7.00s"
    assert "environmental_acoustic_scene" in report
    assert "speaker_diarization_and_emotions" in report


