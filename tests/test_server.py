"""Unit and integration tests for FastAPI and WebSocket streaming server."""
import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from pipeline.server import app
from pipeline.config import SAMPLE_RATE


@pytest.fixture
def client():
    return TestClient(app)


def test_healthcheck_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "hardware_device" in data
    assert "checkpoints" in data
    assert "onnx_engines" in data
    assert data["version"] == "2.0.0"


def test_analyze_audio_multipart(client):
    # Create 2.0-second synthetic sine wave WAV
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, wave, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test_tone.wav", buf, "audio/wav")},
        data={"num_speakers": "1", "apply_noise": "false"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "environmental_acoustic_scene" in data
    assert "speaker_diarization_and_emotions" in data
    assert "original_filename" in data
    assert data["original_filename"] == "test_tone.wav"


def test_websocket_streaming_connection(client):
    with client.websocket_connect("/ws/stream") as websocket:
        init_event = websocket.receive_json()
        assert init_event["event"] == "connected"
        assert init_event["status"] == "ready"
        assert init_event["sample_rate"] == SAMPLE_RATE

        # Send reset action
        websocket.send_json({"action": "reset"})
        reset_ack = websocket.receive_json()
        assert reset_ack["event"] == "reset_acknowledged"

        # Push 0.5s of audio as float32 bytes
        chunk = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        websocket.send_bytes(chunk.tobytes())
