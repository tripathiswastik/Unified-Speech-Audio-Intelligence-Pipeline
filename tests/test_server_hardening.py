"""Security and Hardening Integration Tests for Server and Streaming WebSockets."""
import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import pipeline.server as server_module
from pipeline.server import app, MAX_WS_MESSAGE_BYTES
from pipeline.config import SAMPLE_RATE
from pipeline.realtime_contract import ENCODING_PCM16, ENCODING_FLOAT32, decode_pcm


@pytest.fixture
def client():
    return TestClient(app)


def test_health_without_model_load(client, monkeypatch):
    """Verify /health returns 200 OK without triggering heavy model loading."""
    monkeypatch.setattr(server_module, "_analyzer", None)

    # Calling /health must NOT populate _analyzer
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "hardware_device" in data
    assert server_module._analyzer is None  # Proves non-blocking liveness


def test_readiness_returns_503_when_checkpoints_fail(client, monkeypatch):
    """Verify /ready returns 503 Service Unavailable when model loading fails."""
    monkeypatch.setattr(server_module, "_analyzer", None)

    def mock_get_analyzer():
        raise RuntimeError("Mock failure loading checkpoints")

    monkeypatch.setattr(server_module, "get_analyzer", mock_get_analyzer)

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
    assert data["status"] == "unhealthy"


def test_analyze_empty_file_returns_400(client):
    """Verify uploading an empty file returns HTTP 400."""
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_corrupt_file_clean_error(client):
    """Verify corrupted audio files return clean error messages without leaking internals."""
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("corrupt.wav", b"NOT_A_WAV_HEADER_DATA_12345", "audio/wav")},
        data={"num_speakers": "0"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid or corrupted" in detail
    assert "Traceback" not in detail
    assert "d:\\" not in detail.lower()
    assert "/home" not in detail.lower()


def test_noise_validation_requires_noise_class(client):
    """Verify apply_noise=true without noise_class returns HTTP 422."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", buf, "audio/wav")},
        data={"num_speakers": "0", "apply_noise": "true"},  # missing noise_class
    )
    assert response.status_code == 422
    assert "noise_class is required" in response.json()["detail"]


def test_noise_validation_rejects_out_of_bounds_snr(client):
    """Verify apply_noise=true with out-of-range snr_db returns HTTP 422."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", buf, "audio/wav")},
        data={
            "num_speakers": "0",
            "apply_noise": "true",
            "noise_class": "drilling",
            "snr_db": "100.0",  # Max is 40.0
        },
    )
    assert response.status_code == 422
    assert "snr_db must be a finite float between" in response.json()["detail"]


def test_api_key_authentication_enforcement(client, monkeypatch):
    """Verify API Key enforcement when API_KEY is set in environment."""
    monkeypatch.setattr(server_module, "SERVER_API_KEY", "secret-test-token-12345")

    # Missing API key
    res_no_key = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", b"fake", "audio/wav")},
    )
    assert res_no_key.status_code == 401

    # Invalid API key
    res_bad_key = client.post(
        "/api/v1/analyze",
        files={"file": ("test.wav", b"fake", "audio/wav")},
        headers={"x-api-key": "wrong-token"},
    )
    assert res_bad_key.status_code == 401


def test_websocket_origin_enforcement(client, monkeypatch):
    """Verify WebSocket rejects unauthorized origins with 1008 Policy Violation."""
    monkeypatch.setattr(server_module, "allow_origins", ["http://trusted-site.com"])

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers={"origin": "http://evil-attacker.com"}):
            pass
    assert exc.value.code == 1008


def test_websocket_pcm16_streaming_and_ragged_error(client):
    """Verify PCM16 configuration, streaming, and non-fatal malformed audio reporting."""
    with client.websocket_connect("/ws/stream") as websocket:
        init_event = websocket.receive_json()
        assert init_event["event"] == "connected"
        assert "supported_encodings" in init_event
        assert "pcm16" in init_event["supported_encodings"]

        # 1. Configure to pcm16
        websocket.send_json({"action": "configure", "encoding": "pcm16"})
        conf_ack = websocket.receive_json()
        assert conf_ack["event"] == "configured"
        assert conf_ack["encoding"] == "pcm16"

        # 2. Push valid PCM16 audio bytes (0.2s of int16 zeros)
        int16_chunk = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.int16)
        websocket.send_bytes(int16_chunk.tobytes())

        # 3. Push ragged odd-length bytes (3 bytes = invalid for 16-bit PCM)
        websocket.send_bytes(b"\x00\x01\x02")
        err_event = websocket.receive_json()
        assert err_event["event"] == "error"
        assert err_event["code"] == "malformed_audio"
        assert err_event["fatal"] is False  # Non-fatal error event!


def test_websocket_oversize_frame_rejection(client):
    """Verify message frame exceeding 1 MB is rejected with code 1009."""
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream") as websocket:
            _ = websocket.receive_json()  # connected
            giant_payload = b"\x00" * (MAX_WS_MESSAGE_BYTES + 1024)
            websocket.send_bytes(giant_payload)
            err = websocket.receive_json()
            assert err["event"] == "error"
            assert err["code"] == "too_large"
            # Next read triggers the server-initiated close frame (1009)
            _ = websocket.receive_text()
    assert exc.value.code == 1009


def test_pcm_decoder_unit_contract():
    """Verify realtime_contract.decode_pcm behavior for float32 and pcm16."""
    # Float32 normal
    f32_in = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    f32_out = decode_pcm(f32_in.tobytes(), encoding=ENCODING_FLOAT32)
    np.testing.assert_allclose(f32_in, f32_out)

    # PCM16 normal
    i16_in = np.array([-32768, 0, 16384], dtype=np.int16)
    i16_out = decode_pcm(i16_in.tobytes(), encoding=ENCODING_PCM16)
    expected = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(expected, i16_out, atol=1e-4)

    # Ragged float32
    with pytest.raises(ValueError, match="Ragged byte length"):
        decode_pcm(b"\x00\x00\x00", encoding=ENCODING_FLOAT32)

    # Ragged pcm16
    with pytest.raises(ValueError, match="Ragged byte length"):
        decode_pcm(b"\x00\x00\x00", encoding=ENCODING_PCM16)

    # NaN floats rejected
    nan_bytes = np.array([np.nan, 0.0], dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="non-finite values"):
        decode_pcm(nan_bytes, encoding=ENCODING_FLOAT32)
