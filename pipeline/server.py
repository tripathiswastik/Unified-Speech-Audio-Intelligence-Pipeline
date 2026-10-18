"""FastAPI & WebSocket Streaming Server for Unified Audio Intelligence.

Provides:
  1. GET  /health          - Non-blocking liveness probe (status, hardware engine, demo mode).
  2. GET  /ready           - Readiness probe verifying loaded models (returns 200 or 503).
  3. POST /api/v1/analyze  - Safe synchronous multipart audio analysis with pre-decode checks.
  4. WS   /ws/stream       - Hardened real-time bidirectional audio streaming for microphones/WebSockets.
"""
import io
import os
import secrets
import logging
import threading
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import soundfile as sf
import torch
import librosa
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Header,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from pipeline.config import CHECKPOINTS_DIR, SAMPLE_RATE, CLASSES
from pipeline.device import get_device
from pipeline.streaming import StreamingDiarizer
from pipeline.unified_analyzer import UnifiedAudioAnalyzer
from pipeline.realtime_contract import (
    ENCODING_FLOAT32,
    ENCODING_PCM16,
    SUPPORTED_ENCODINGS,
    decode_pcm,
    event_payload,
    error_payload,
)

logger = logging.getLogger(__name__)

# Resource & Rate Limits
MAX_UPLOAD_BYTES = 25 * 1024 * 1024       # 25 MB file upload limit
MAX_DECODED_AUDIO_BYTES = 256 * 1024 * 1024 # 256 MiB uncompressed limit
MAX_AUDIO_DURATION_SECONDS = 300.0        # 5 minutes duration limit
MAX_WS_MESSAGE_BYTES = 1 * 1024 * 1024    # 1 MiB max WebSocket message frame
MAX_ACTIVE_WS_CONNECTIONS = 8             # Max concurrent active WebSocket clients
MAX_WS_SESSION_AUDIO_SECONDS = 300.0      # 5 minutes per WebSocket session
WS_IDLE_TIMEOUT_SECONDS = 60.0            # 60s idle timeout for sockets

# Security & Auth Configuration
SERVER_API_KEY = os.getenv("API_KEY")
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allow_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def origin_ok(origin: Optional[str]) -> bool:
    """Validate WebSocket origin header against allowed origins."""
    if not origin:
        return True  # Allow non-browser clients (CLI, test suites, curl)
    origin_clean = origin.rstrip("/")
    if "*" in allow_origins:
        return True
    for o in allow_origins:
        if o.rstrip("/") == origin_clean:
            return True
    # Always permit localhost/127.0.0.1 origins
    if origin_clean.startswith("http://localhost") or origin_clean.startswith("http://127.0.0.1"):
        return True
    return False


def api_key_ok(key: Optional[str]) -> bool:
    """Verify optional API Key header in constant time."""
    if not SERVER_API_KEY:
        return True
    if not key:
        return False
    return secrets.compare_digest(key, SERVER_API_KEY)


def verify_api_key_header(x_api_key: Optional[str] = Header(None)) -> None:
    """FastAPI dependency for REST endpoints requiring API Key."""
    if not api_key_ok(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API Key.",
        )


# Global Thread-Safe Analyzer Singleton & Locks
_analyzer: Optional[UnifiedAudioAnalyzer] = None
_analyzer_lock = threading.Lock()
_inference_lock = asyncio.Lock()
_active_ws_connections: int = 0
_ws_connection_lock = threading.Lock()


def get_analyzer() -> UnifiedAudioAnalyzer:
    """Thread-safe double-checked singleton accessor for UnifiedAudioAnalyzer."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                require_trained = (
                    os.getenv("REQUIRE_TRAINED_CHECKPOINTS", "0").strip().lower()
                    in ("1", "true", "yes")
                )
                try:
                    _analyzer = UnifiedAudioAnalyzer(strict_checkpoints=True, demo_mode=False)
                    logger.info("Loaded trained checkpoints in strict mode.")
                except Exception as e:
                    if require_trained:
                        logger.exception("Trained checkpoints are required but failed to load.")
                        raise RuntimeError(
                            f"Production server refused to serve inference: Trained checkpoints are strictly required: {e}"
                        ) from e
                    logger.warning(
                        "Failed to load trained checkpoints (%s). Falling back to demo_mode=True.", e
                    )
                    _analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=True)
                    logger.info("Initialized analyzer in demo_mode=True (synthetic checkpoints).")
    return _analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for background model preloading at startup."""
    try:
        await run_in_threadpool(get_analyzer)
        logger.info("Lifespan startup: Analyzer preloaded successfully.")
    except Exception as e:
        logger.warning("Lifespan startup: Analyzer preloading deferred or failed: %s", e)
    yield


app = FastAPI(
    title="UrbanSound Speech & Audio Intelligence API",
    description="Multi-task streaming audio understanding: Speaker Diarization, Verification, Emotion, and Noise-Invariant Sensing.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: Only allow credentials when there is NO wildcard origin
allow_credentials = "*" not in allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck():
    """Non-blocking liveness probe. Inspects server state without triggering heavy model loads."""
    device = get_device()
    global _analyzer
    is_demo = _analyzer.demo_mode if _analyzer is not None else False
    return {
        "status": "healthy",
        "hardware_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "demo_mode": is_demo,
        "version": "2.0.0",
        "checkpoints": {
            "urbansound_cnn": {
                "loaded": _analyzer.urban_checkpoint_loaded if _analyzer else False,
                "sha256": _analyzer.urban_checkpoint_sha256 if _analyzer else None,
            },
            "emotion_ser": {
                "loaded": _analyzer.emotion_checkpoint_loaded if _analyzer else False,
                "sha256": _analyzer.emotion_checkpoint_sha256 if _analyzer else None,
            },
            "speaker_id": {
                "loaded": _analyzer.speaker_checkpoint_loaded if _analyzer else False,
                "sha256": _analyzer.speaker_checkpoint_sha256 if _analyzer else None,
            },
        },
        "onnx_engines": {
            "se_audiocnn_onnx": (CHECKPOINTS_DIR / "se_audiocnn.onnx").exists(),
            "se_speaker_embedding_onnx": (CHECKPOINTS_DIR / "se_speaker_embedding.onnx").exists(),
        },
    }


@app.get("/ready")
async def readiness_probe():
    """Readiness probe. Returns 200 when models are loaded, or 503 Service Unavailable on failure."""
    try:
        ana = await run_in_threadpool(get_analyzer)
        return {
            "ready": True,
            "demo_mode": ana.demo_mode,
            "checkpoints": {
                "urbansound_cnn": {
                    "loaded": ana.urban_checkpoint_loaded,
                    "sha256": ana.urban_checkpoint_sha256,
                },
                "emotion_ser": {
                    "loaded": ana.emotion_checkpoint_loaded,
                    "sha256": ana.emotion_checkpoint_sha256,
                },
                "speaker_id": {
                    "loaded": ana.speaker_checkpoint_loaded,
                    "sha256": ana.speaker_checkpoint_sha256,
                },
            },
        }
    except Exception as e:
        logger.warning("Readiness probe check failed: %s", e)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "status": "unhealthy",
                "detail": "Model checkpoints are not ready to serve traffic.",
            },
        )


def prepare_waveform(content: bytes) -> Tuple[np.ndarray, int]:
    """Inspect audio header, validate limits before full decoding, downmix to mono, and resample."""
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # 1. Header pre-inspection (prevent decompression bombs)
    try:
        info = sf.info(io.BytesIO(content))
    except Exception as err:
        logger.warning("Audio header inspection failed: %s", err)
        raise HTTPException(status_code=400, detail="Invalid or corrupted audio file.") from err

    if info.samplerate <= 0:
        raise HTTPException(status_code=400, detail="Invalid audio sample rate in header.")
    if info.channels <= 0 or info.channels > 8:
        raise HTTPException(status_code=400, detail=f"Unsupported channel count ({info.channels}). Must be 1 to 8.")
    if info.frames <= 0:
        raise HTTPException(status_code=400, detail="Audio file contains 0 audio frames.")

    decoded_size = info.frames * info.channels * 4  # float32
    if decoded_size > MAX_DECODED_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Decoded audio size ({decoded_size / (1024*1024):.1f} MB) exceeds maximum limit (256 MB).",
        )

    duration_sec = info.frames / float(info.samplerate)
    if duration_sec > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({duration_sec:.1f}s) exceeds the maximum limit of {MAX_AUDIO_DURATION_SECONDS}s.",
        )

    # 2. Decode samples
    try:
        raw_samples, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as dec_err:
        logger.warning("Audio decoding failed: %s", dec_err)
        raise HTTPException(status_code=400, detail="Failed to decode audio samples.") from dec_err

    # 3. Correct mono downmix: average across channels (axis 1)
    if raw_samples.shape[1] == 1:
        waveform = raw_samples[:, 0]
    else:
        waveform = np.mean(raw_samples, axis=1)

    if not np.all(np.isfinite(waveform)):
        raise HTTPException(status_code=400, detail="Audio waveform contains non-finite values (NaN or Inf).")

    # 4. Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    return waveform, SAMPLE_RATE


@app.post("/api/v1/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    num_speakers: int = Form(0, ge=0, le=8),
    apply_noise: bool = Form(False),
    noise_class: Optional[str] = Form(None),
    snr_db: float = Form(10.0),
    x_api_key: Optional[str] = Header(None),
):
    """Analyze uploaded audio file using the unified multi-task intelligence pipeline."""
    verify_api_key_header(x_api_key)

    # 1. Enforce payload size limit in bounded chunks
    CHUNK_SIZE = 64 * 1024
    total_bytes = 0
    buffer = io.BytesIO()
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Payload Too Large: File exceeded 25 MB limit ({total_bytes / (1024 * 1024):.1f} MB uploaded).",
            )
        buffer.write(chunk)
    content = buffer.getvalue()

    # 2. Validate noise parameters strictly
    if apply_noise:
        if not noise_class:
            raise HTTPException(status_code=422, detail="noise_class is required when apply_noise is enabled.")
        if noise_class not in CLASSES:
            raise HTTPException(status_code=422, detail=f"Invalid noise_class '{noise_class}'. Must be one of: {CLASSES}")
        if not np.isfinite(snr_db) or snr_db < -20.0 or snr_db > 40.0:
            raise HTTPException(status_code=422, detail="snr_db must be a finite float between -20.0 and 40.0 dB.")

    # 3. Offload waveform decoding and preparation to threadpool
    waveform, _ = await run_in_threadpool(prepare_waveform, content)

    # 4. Apply noise injection if requested (in threadpool)
    achieved_snr = None
    if apply_noise and noise_class:
        def _apply_noise_step():
            from pipeline.urban_noise_mixer import UrbanNoiseMixer
            mixer = UrbanNoiseMixer(classes=[noise_class])
            return mixer.mix(waveform, snr_db=snr_db, noise_class=noise_class)

        try:
            waveform, chosen_class, achieved_snr = await run_in_threadpool(_apply_noise_step)
        except ValueError as val_err:
            logger.warning("Noise injection invalid input: %s", val_err)
            raise HTTPException(status_code=400, detail="Urban noise mixing parameter error.") from val_err
        except Exception as mix_err:
            logger.exception("Unexpected error during noise injection: %s", mix_err)
            raise HTTPException(status_code=500, detail="Urban noise mixing failed.") from mix_err

    # 5. Run unified analysis under inference lock in threadpool
    try:
        ana = await run_in_threadpool(get_analyzer)
        speakers_arg = num_speakers if num_speakers > 0 else None

        async with _inference_lock:
            report = await run_in_threadpool(
                ana.analyze_waveform,
                waveform,
                file_name=file.filename or "uploaded_audio.wav",
                num_speakers=speakers_arg,
            )

        report["original_filename"] = file.filename
        report["demo_mode"] = ana.demo_mode
        if achieved_snr is not None:
            report["noise_injection"] = {
                "noise_class": noise_class,
                "target_snr_db": snr_db,
                "achieved_snr_db": achieved_snr,
            }
        return report
    except Exception as ana_err:
        logger.exception("Audio intelligence analysis failed: %s", ana_err)
        raise HTTPException(status_code=500, detail="Audio intelligence analysis failed.") from ana_err


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Hardened bidirectional WebSocket streaming endpoint for low-latency live microphone ingestion."""
    # 1. Security Check: Origin Verification
    origin = websocket.headers.get("origin")
    if not origin_ok(origin):
        logger.warning("Rejected WebSocket connection from unauthorized origin: %s", origin)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Security Check: API Key Verification
    api_key = websocket.headers.get("x-api-key")
    if not api_key_ok(api_key):
        logger.warning("Rejected WebSocket connection due to missing or invalid API Key.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Concurrency Limit
    global _active_ws_connections
    with _ws_connection_lock:
        if _active_ws_connections >= MAX_ACTIVE_WS_CONNECTIONS:
            logger.warning("WebSocket connection rejected: maximum connection limit (%d) reached.", MAX_ACTIVE_WS_CONNECTIONS)
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return
        _active_ws_connections += 1

    await websocket.accept()

    try:
        ana = await run_in_threadpool(get_analyzer)
        speaker_model = getattr(ana.speaker_identifier, "model", None)
        diarizer = StreamingDiarizer(
            embedding_net=speaker_model,
            buffer_duration=2.0,
            hop_duration=0.5,
            sr=SAMPLE_RATE,
            energy_threshold=0.015,
            similarity_threshold=0.65,
        )

        current_encoding = ENCODING_FLOAT32
        total_audio_seconds: float = 0.0

        await websocket.send_json({
            "event": "connected",
            "sample_rate": SAMPLE_RATE,
            "buffer_duration": 2.0,
            "hop_duration": 0.5,
            "supported_encodings": list(SUPPORTED_ENCODINGS),
            "default_encoding": ENCODING_FLOAT32,
            "demo_mode": ana.demo_mode,
            "status": "ready",
        })

        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=WS_IDLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.info("Closing idle WebSocket connection after %ds timeout.", WS_IDLE_TIMEOUT_SECONDS)
                await websocket.send_json(error_payload("idle_timeout", "Session timed out due to inactivity.", fatal=True))
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                break
            except (WebSocketDisconnect, RuntimeError):
                break

            if message.get("type") == "websocket.disconnect":
                break

            pcm_chunk: Optional[np.ndarray] = None

            # Binary payload
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if len(raw_bytes) > MAX_WS_MESSAGE_BYTES:
                    await websocket.send_json(error_payload("too_large", "Message frame exceeded 1 MB limit.", fatal=True))
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    break

                try:
                    pcm_chunk = decode_pcm(raw_bytes, encoding=current_encoding)
                except ValueError as val_err:
                    await websocket.send_json(error_payload("malformed_audio", str(val_err), fatal=False))
                    continue

            # Text / JSON payload
            elif "text" in message and message["text"]:
                raw_text = message["text"]
                if len(raw_text.encode("utf-8")) > MAX_WS_MESSAGE_BYTES:
                    await websocket.send_json(error_payload("too_large", "Message frame exceeded 1 MB limit.", fatal=True))
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    break

                import json
                try:
                    data = json.loads(raw_text)
                except Exception:
                    await websocket.send_json(error_payload("invalid_json", "Failed to parse JSON text message.", fatal=False))
                    continue

                action = data.get("action", "push_audio")

                if action == "configure":
                    requested_enc = data.get("encoding", "").lower().strip()
                    if requested_enc in SUPPORTED_ENCODINGS:
                        current_encoding = requested_enc
                        await websocket.send_json({"event": "configured", "encoding": current_encoding})
                    else:
                        await websocket.send_json(error_payload("unsupported_encoding", f"Encoding must be one of {SUPPORTED_ENCODINGS}", fatal=False))
                    continue

                elif action == "reset":
                    diarizer.reset()
                    total_audio_seconds = 0.0
                    await websocket.send_json({"event": "reset_acknowledged"})
                    continue

                elif action == "push_audio":
                    import base64
                    b64_audio = data.get("audio_b64", "")
                    if b64_audio:
                        try:
                            raw = base64.b64decode(b64_audio)
                            enc = data.get("encoding", current_encoding)
                            pcm_chunk = decode_pcm(raw, encoding=enc)
                        except ValueError as val_err:
                            await websocket.send_json(error_payload("malformed_audio", str(val_err), fatal=False))
                            continue
                        except Exception:
                            await websocket.send_json(error_payload("invalid_base64", "Failed to decode base64 audio payload.", fatal=False))
                            continue

            # Process valid audio chunk
            if pcm_chunk is not None and len(pcm_chunk) > 0:
                chunk_duration = len(pcm_chunk) / float(SAMPLE_RATE)
                total_audio_seconds += chunk_duration
                if total_audio_seconds > MAX_WS_SESSION_AUDIO_SECONDS:
                    await websocket.send_json(error_payload("budget_exceeded", "Session audio budget exceeded (max 300s).", fatal=True))
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    break

                # Run diarizer under inference lock in threadpool
                async with _inference_lock:
                    events = await run_in_threadpool(diarizer.push_audio, pcm_chunk)

                for ev in events:
                    await websocket.send_json(event_payload(ev))

    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as ex:
        logger.exception("Unexpected error in WebSocket session: %s", ex)
    finally:
        with _ws_connection_lock:
            _active_ws_connections = max(0, _active_ws_connections - 1)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Launch the Uvicorn ASGI server with configured logging."""
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Launching Real-Time Audio Intelligence API on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
