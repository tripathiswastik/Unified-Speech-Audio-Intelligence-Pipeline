"""Integration tests for live WebSocket client against a real running Uvicorn server."""
import time
import socket
import threading
import json
import pytest
import numpy as np
import uvicorn
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

import pipeline.server as server_module
from pipeline.server import app
from pipeline.config import SAMPLE_RATE


def get_free_port() -> int:
    """Acquire an ephemeral free port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Spin up a real background Uvicorn server for live socket tests."""
    port = get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"ws://127.0.0.1:{port}/ws/stream"
    # Poll until server is ready
    for _ in range(50):
        try:
            with connect(url, close_timeout=1.0) as ws:
                _ = ws.recv()
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Live Uvicorn server failed to accept connections within 5 seconds.")

    yield url

    server.should_exit = True
    thread.join(timeout=3.0)


def test_live_handshake(live_server):
    """Verify live client handshake receives connected event and metadata."""
    with connect(live_server, close_timeout=2.0) as ws:
        msg = json.loads(ws.recv())
        assert msg["event"] == "connected"
        assert msg["status"] == "ready"
        assert msg["sample_rate"] == SAMPLE_RATE
        assert "supported_encodings" in msg
        assert "pcm16" in msg["supported_encodings"]


def test_live_pcm16_chunk_streaming_roundtrip(live_server):
    """Stream 5 sequential PCM16 chunks to live socket and verify roundtrip."""
    with connect(live_server, close_timeout=2.0) as ws:
        _ = ws.recv()  # connected

        # 1. Configure encoding
        ws.send(json.dumps({"action": "configure", "encoding": "pcm16"}))
        ack = json.loads(ws.recv())
        assert ack["event"] == "configured"
        assert ack["encoding"] == "pcm16"

        # 2. Stream 5 sequential 250ms chunks of synthetic tone audio
        chunk_samples = int(0.25 * SAMPLE_RATE)
        t = np.linspace(0, 0.25, chunk_samples, endpoint=False)
        chunk_f32 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        chunk_pcm16 = (chunk_f32 * 32767).astype(np.int16).tobytes()

        for _ in range(5):
            ws.send(chunk_pcm16)

        # 3. Request reset
        ws.send(json.dumps({"action": "reset"}))
        reset_ack = json.loads(ws.recv())
        assert reset_ack["event"] == "reset_acknowledged"


def test_live_rejects_unauthorized_origin(live_server, monkeypatch):
    """Verify live server actively rejects connection when origin is unauthorized."""
    monkeypatch.setattr(server_module, "allow_origins", ["http://trusted-site.com"])

    with pytest.raises((ConnectionClosed, InvalidStatus)) as exc:
        with connect(
            live_server,
            additional_headers={"Origin": "http://malicious-site.com"},
            close_timeout=2.0,
        ) as ws:
            _ = ws.recv()

    if isinstance(exc.value, InvalidStatus):
        assert exc.value.response.status_code == 403
    else:
        assert exc.value.rcvd.code == 1008


def test_live_rejects_invalid_api_key(live_server, monkeypatch):
    """Verify live server closes connection when API_KEY is set and client provides invalid key."""
    monkeypatch.setattr(server_module, "SERVER_API_KEY", "super-secret-key-999")

    with pytest.raises((ConnectionClosed, InvalidStatus)) as exc:
        with connect(
            live_server,
            additional_headers={"X-API-Key": "wrong-key"},
            close_timeout=2.0,
        ) as ws:
            _ = ws.recv()

    if isinstance(exc.value, InvalidStatus):
        assert exc.value.response.status_code == 403
    else:
        assert exc.value.rcvd.code == 1008
