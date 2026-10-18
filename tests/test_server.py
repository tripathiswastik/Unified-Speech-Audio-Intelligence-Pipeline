"""Unit and integration tests for FastAPI and WebSocket streaming server."""
import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import pipeline.server as server_module
from pipeline.server import app, MAX_UPLOAD_BYTES, MAX_WS_MESSAGE_BYTES
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


def test_readiness_probe_endpoint(client):
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "ready" in data
    assert "checkpoints" in data
    assert "urbansound_cnn" in data["checkpoints"]


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


def test_analyze_audio_with_noise_injection(client):
    """Verify that enabling apply_noise with a valid noise_class injects noise and reports SNR."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, wave, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("speech_noisy.wav", buf, "audio/wav")},
        data={
            "num_speakers": "1",
            "apply_noise": "true",
            "noise_class": "siren",
            "snr_db": "10.0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "noise_injection" in data
    assert data["noise_injection"]["noise_class"] == "siren"
    assert "achieved_snr_db" in data["noise_injection"]


def test_payload_too_large(client, monkeypatch):
    """Verify that uploading a payload exceeding MAX_UPLOAD_BYTES returns HTTP 413."""
    import pipeline.server
    monkeypatch.setattr(pipeline.server, "MAX_UPLOAD_BYTES", 1024)
    oversized_data = b"0" * 2048
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("giant_file.wav", oversized_data, "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert response.status_code == 413
    assert "Payload Too Large" in response.json()["detail"]


def test_invalid_audio_format(client):
    """Verify that uploading malformed audio bytes returns HTTP 400."""
    corrupted_bytes = b"RIFF0000WAVEfmt corrupted non audio header data"
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("corrupt.wav", corrupted_bytes, "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert response.status_code == 400


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


def test_analyzer_fallback_when_checkpoints_missing(monkeypatch):
    """Verify that get_analyzer() falls back to demo_mode=True without NameError when checkpoints are missing."""
    import pipeline.server
    monkeypatch.setattr(pipeline.server, "_analyzer", None)
    monkeypatch.setenv("REQUIRE_TRAINED_CHECKPOINTS", "0")

    original_init = pipeline.server.UnifiedAudioAnalyzer.__init__

    def mock_init(self, strict_checkpoints=True, demo_mode=False, **kwargs):
        if strict_checkpoints and not demo_mode:
            raise FileNotFoundError("Mock missing checkpoints")
        original_init(self, strict_checkpoints=False, demo_mode=True, **kwargs)

    monkeypatch.setattr(pipeline.server.UnifiedAudioAnalyzer, "__init__", mock_init)

    ana = pipeline.server.get_analyzer()
    assert ana is not None
    assert ana.demo_mode is True


def test_analyzer_strict_mode_enforcement(monkeypatch):
    """Verify that get_analyzer() raises RuntimeError if REQUIRE_TRAINED_CHECKPOINTS=1 and checkpoints fail."""
    import pipeline.server
    monkeypatch.setattr(pipeline.server, "_analyzer", None)
    monkeypatch.setenv("REQUIRE_TRAINED_CHECKPOINTS", "1")

    def mock_init(self, strict_checkpoints=True, demo_mode=False, **kwargs):
        raise FileNotFoundError("Mock missing checkpoints")

    monkeypatch.setattr(pipeline.server.UnifiedAudioAnalyzer, "__init__", mock_init)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.server.get_analyzer()
    assert "Trained checkpoints are strictly required" in str(exc_info.value)


def test_analyze_empty_file_returns_400(client):
    """Verify uploading an empty file (0 bytes) returns HTTP 400."""
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_noise_validation_missing_or_invalid_class(client):
    """Verify noise validation rejects missing or non-existent noise classes."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    # Missing class
    res1 = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", buf.getvalue(), "audio/wav")},
        data={"apply_noise": "true"},
    )
    assert res1.status_code == 422
    assert "noise_class is required" in res1.json()["detail"]

    # Invalid class
    res2 = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", buf.getvalue(), "audio/wav")},
        data={"apply_noise": "true", "noise_class": "invalid_noise_class_xyz"},
    )
    assert res2.status_code == 422
    assert "Invalid noise_class" in res2.json()["detail"]


def test_api_key_authentication_enforcement(client, monkeypatch):
    """Verify REST endpoint rejects requests when API Key is invalid or missing."""
    monkeypatch.setattr(server_module, "SERVER_API_KEY", "test-secret-key-123")

    # Missing API Key
    res_no_key = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", b"dummy", "audio/wav")},
    )
    assert res_no_key.status_code == 401
    assert "Unauthorized" in res_no_key.json()["detail"]

    # Invalid API Key
    res_bad_key = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", b"dummy", "audio/wav")},
        headers={"x-api-key": "wrong-key"},
    )
    assert res_bad_key.status_code == 401


def test_websocket_pcm16_streaming_and_malformed_handling(client):
    """Verify PCM16 configuration, streaming, and non-fatal malformed audio handling."""
    with client.websocket_connect("/ws/stream") as websocket:
        init_event = websocket.receive_json()
        assert init_event["event"] == "connected"

        # Configure to PCM16
        websocket.send_json({"action": "configure", "encoding": "pcm16"})
        conf = websocket.receive_json()
        assert conf["event"] == "configured"
        assert conf["encoding"] == "pcm16"

        # Stream valid PCM16 chunk
        valid_pcm16 = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.int16).tobytes()
        websocket.send_bytes(valid_pcm16)

        # Send malformed ragged byte chunk (3 bytes: invalid for 16-bit PCM)
        websocket.send_bytes(b"\x01\x02\x03")
        err = websocket.receive_json()
        assert err["event"] == "error"
        assert err["code"] == "malformed_audio"
        assert err["fatal"] is False


def test_websocket_unauthorized_origin_rejection(client, monkeypatch):
    """Verify WebSocket rejects connection from unauthorized origin."""
    monkeypatch.setattr(server_module, "allow_origins", ["http://trusted-site.com"])

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers={"origin": "http://untrusted-attacker.com"}):
            pass
    assert exc.value.code == 1008


def test_websocket_oversize_message_rejection(client):
    """Verify WebSocket rejects message frames exceeding MAX_WS_MESSAGE_BYTES."""
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream") as websocket:
            _ = websocket.receive_json()
            oversized_frame = b"\x00" * (MAX_WS_MESSAGE_BYTES + 1024)
            websocket.send_bytes(oversized_frame)
            err = websocket.receive_json()
            assert err["event"] == "error"
            assert err["code"] == "too_large"
            _ = websocket.receive_text()
    assert exc.value.code == 1009

