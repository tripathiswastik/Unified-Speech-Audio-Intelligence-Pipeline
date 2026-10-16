"""FastAPI & WebSocket Streaming Server for Unified Audio Intelligence.
Provides:
  1. GET  /health          - Hardware engine status, checkpoint integrity & ONNX status.
  2. GET  /ready           - Readiness check verifying all trained model checkpoints are loaded.
  3. POST /api/v1/analyze  - Synchronous multipart/form-data audio analysis with file-size limits.
  4. WS   /ws/stream       - Low-latency real-time bidirectional audio streaming for microphones/WebSockets.
"""
import io
import os
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.config import BASE_DIR, CHECKPOINTS_DIR, SAMPLE_RATE, CLASSES
from pipeline.device import get_device
from pipeline.streaming import StreamingPCEN, StreamingDiarizer, StreamingSpeechEvent
from pipeline.unified_analyzer import UnifiedAudioAnalyzer

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB file size limit

app = FastAPI(
    title="UrbanSound Speech & Audio Intelligence API",
    description="Multi-task streaming audio understanding: Speaker Diarization, Verification, Emotion, and Noise-Invariant Sensing.",
    version="2.0.0",
)

# Secure CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allow_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global cached analyzer instance
_analyzer: Optional[UnifiedAudioAnalyzer] = None


MAX_AUDIO_DURATION_SECONDS = 300.0  # 5 minutes limit


def get_analyzer() -> UnifiedAudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        require_trained = os.getenv("REQUIRE_TRAINED_CHECKPOINTS", "0").strip().lower() in ("1", "true", "yes")
        try:
            # First attempt strict checkpoint loading
            _analyzer = UnifiedAudioAnalyzer(strict_checkpoints=True, demo_mode=False)
        except Exception as e:
            if require_trained:
                raise RuntimeError(
                    f"Production server refused to serve inference: Trained checkpoints are strictly required: {e}"
                )
            logger.warning("Checkpoints missing or incomplete; falling back to explicit demo_mode=True: %s", e)
            _analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=True)
    return _analyzer


class HealthResponse(BaseModel):
    status: str
    hardware_device: str
    cuda_available: bool
    demo_mode: bool
    checkpoints: Dict[str, bool]
    onnx_engines: Dict[str, bool]
    version: str


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    """Liveness probe: verifies system health, GPU hardware acceleration, and model provenance."""
    dev = get_device(verbose=False)
    ana = get_analyzer()
    
    onnx_audio = (CHECKPOINTS_DIR / "se_audiocnn.onnx").exists()
    onnx_spk = (CHECKPOINTS_DIR / "se_speaker_embedding.onnx").exists()
    
    return HealthResponse(
        status="healthy",
        hardware_device=str(dev),
        cuda_available=torch.cuda.is_available(),
        demo_mode=ana.demo_mode,
        checkpoints={
            "urbansound_cnn": ana.urban_checkpoint_loaded,
            "emotion_ser": ana.emotion_checkpoint_loaded,
            "speaker_id": ana.speaker_checkpoint_loaded,
        },
        onnx_engines={
            "se_audiocnn_onnx": onnx_audio,
            "se_speaker_embedding_onnx": onnx_spk,
        },
        version="2.0.0",
    )


@app.get("/ready")
def readiness_check():
    """Readiness probe: returns 200 OK only when all required trained checkpoints are loaded."""
    ana = get_analyzer()
    all_ready = (
        ana.urban_checkpoint_loaded and
        ana.emotion_checkpoint_loaded and
        ana.speaker_checkpoint_loaded
    )
    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": all_ready,
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
        },
    )


@app.post("/api/v1/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    num_speakers: int = Form(0),
    apply_noise: bool = Form(False),
    noise_class: Optional[str] = Form(None),
    snr_db: float = Form(10.0),
):
    """Analyze uploaded audio file using the unified multi-task intelligence pipeline."""
    # 1. Enforce payload size limit in bounded chunks (prevents allocating unbounded memory upfront)
    CHUNK_SIZE = 64 * 1024  # 64 KB
    total_bytes = 0
    buffer = io.BytesIO()
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Payload Too Large: File exceeded 25 MB limit ({total_bytes / (1024 * 1024):.1f} MB uploaded).",
            )
        buffer.write(chunk)
    content = buffer.getvalue()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # 2. Decode audio waveform
    try:
        audio_io = io.BytesIO(content)
        waveform, sr = sf.read(audio_io, dtype="float32")
    except Exception as dec_err:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio file: {str(dec_err)}")

    if len(waveform) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file contains 0 audio samples.")

    # Validate finite values
    if not np.all(np.isfinite(waveform)):
        raise HTTPException(status_code=400, detail="Audio waveform contains non-finite values (NaN or Inf).")

    # Validate maximum duration (300 seconds)
    duration_sec = len(waveform) / float(sr) if sr > 0 else 0.0
    if duration_sec > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({duration_sec:.1f}s) exceeds the maximum limit of {MAX_AUDIO_DURATION_SECONDS}s.",
        )

    # Convert multi-channel to mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1 if waveform.shape[1] < waveform.shape[0] else 0)

    # Resample if sr != SAMPLE_RATE
    if sr != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    # 3. Apply noise injection if requested
    achieved_snr = None
    if apply_noise and noise_class:
        if noise_class not in CLASSES:
            raise HTTPException(status_code=422, detail=f"Invalid noise_class '{noise_class}'. Must be one of: {CLASSES}")
        try:
            from pipeline.urban_noise_mixer import UrbanNoiseMixer
            mixer = UrbanNoiseMixer(classes=[noise_class])
            waveform, chosen_class, achieved_snr = mixer.mix(waveform, snr_db=snr_db, noise_class=noise_class)
        except ValueError as val_err:
            logger.warning("Noise injection invalid input: %s", val_err)
            raise HTTPException(status_code=400, detail=f"Urban noise mixing parameter error: {str(val_err)}")
        except Exception as mix_err:
            logger.exception("Unexpected error during noise injection: %s", mix_err)
            raise HTTPException(status_code=500, detail=f"Urban noise mixing failed: {str(mix_err)}")

    # 4. Run unified analysis
    try:
        ana = get_analyzer()
        report = ana.analyze_waveform(
            waveform,
            file_name=file.filename or "uploaded_audio.wav",
            num_speakers=num_speakers if num_speakers > 0 else 2,
        )
        report["original_filename"] = file.filename
        if achieved_snr is not None:
            report["noise_injection"] = {
                "noise_class": noise_class,
                "target_snr_db": snr_db,
                "achieved_snr_db": achieved_snr,
            }
        return report
    except Exception as ana_err:
        raise HTTPException(status_code=500, detail=f"Audio intelligence analysis failed: {str(ana_err)}")


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket streaming endpoint for low-latency live microphone ingestion.
    
    Supported Inbound Payloads:
      1. Binary PCM: Float32 array bytes at 22,050 Hz or 16,000 Hz.
      2. JSON Text: {"action": "push_audio", "audio_b64": "<base64_pcm_float32>"}
      3. JSON Reset: {"action": "reset"}
    
    Outbound Payloads:
      JSON message whenever a speech turn boundary or silence is detected:
      {
        "event": "speech_segment",
        "event_type": "window" | "turn_start" | "turn_complete" | "speaker_shift",
        "timestamp_start": 0.0,
        "timestamp_end": 0.5,
        "speaker_tag": "Speaker_1",
        "confidence": 0.94,
        "is_speech": true
      }
    """
    await websocket.accept()
    
    # Reuse the analyzer's loaded speaker checkpoint for proven weights
    ana = get_analyzer()
    speaker_model = getattr(ana.speaker_identifier, "model", None)
    
    diarizer = StreamingDiarizer(
        embedding_net=speaker_model,
        buffer_duration=2.0,
        hop_duration=0.5,
        sr=SAMPLE_RATE,
        energy_threshold=0.015,
        similarity_threshold=0.65,
    )
    pcen = StreamingPCEN(n_mels=128)
    
    await websocket.send_json({
        "event": "connected",
        "sample_rate": SAMPLE_RATE,
        "buffer_duration": 2.0,
        "hop_duration": 0.5,
        "status": "ready"
    })
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if message.get("type") == "websocket.disconnect":
                break
            
            # 1. Binary payload (direct raw float32 PCM bytes)
            if "bytes" in message and message["bytes"]:
                pcm_bytes = message["bytes"]
                pcm_chunk = np.frombuffer(pcm_bytes, dtype=np.float32)
                events = diarizer.push_audio(pcm_chunk)
                for ev in events:
                    await websocket.send_json({
                        "event": "speech_segment",
                        "event_type": getattr(ev, "event_type", "window"),
                        "timestamp_start": ev.timestamp_start,
                        "timestamp_end": ev.timestamp_end,
                        "speaker_tag": ev.speaker_tag,
                        "confidence": ev.confidence,
                        "is_speech": ev.is_speech,
                    })
                    
            # 2. Text payload (JSON command or base64 audio chunk)
            elif "text" in message and message["text"]:
                import json
                try:
                    data = json.loads(message["text"])
                    action = data.get("action", "push_audio")
                    
                    if action == "reset":
                        diarizer.reset()
                        pcen.reset()
                        await websocket.send_json({"event": "reset_acknowledged"})
                    elif action == "push_audio":
                        b64_audio = data.get("audio_b64", "")
                        if b64_audio:
                            raw = base64.b64decode(b64_audio)
                            pcm_chunk = np.frombuffer(raw, dtype=np.float32)
                            events = diarizer.push_audio(pcm_chunk)
                            for ev in events:
                                await websocket.send_json({
                                    "event": "speech_segment",
                                    "event_type": getattr(ev, "event_type", "window"),
                                    "timestamp_start": ev.timestamp_start,
                                    "timestamp_end": ev.timestamp_end,
                                    "speaker_tag": ev.speaker_tag,
                                    "confidence": ev.confidence,
                                    "is_speech": ev.is_speech,
                                })
                except Exception as ex:
                    await websocket.send_json({"event": "error", "message": str(ex)})
                    
    except (WebSocketDisconnect, RuntimeError):
        pass


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Launch the Uvicorn ASGI server."""
    import uvicorn
    print(f"\n[FastAPI Server] Launching Real-Time Audio Intelligence API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
