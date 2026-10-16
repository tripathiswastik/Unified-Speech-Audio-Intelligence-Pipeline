"""FastAPI & WebSocket Streaming Server for Unified Audio Intelligence.
Provides:
  1. GET  /health          - Hardware engine status, checkpoint integrity & ONNX status.
  2. POST /api/v1/analyze  - Synchronous multipart/form-data audio analysis.
  3. WS   /ws/stream       - Low-latency real-time bidirectional audio streaming for microphones/WebSockets.
"""
import io
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.config import BASE_DIR, CHECKPOINTS_DIR, SAMPLE_RATE, CLASSES
from pipeline.device import get_device
from pipeline.streaming import StreamingPCEN, StreamingDiarizer, StreamingSpeechEvent
from pipeline.unified_analyzer import UnifiedAudioAnalyzer

app = FastAPI(
    title="UrbanSound Speech & Audio Intelligence API",
    description="Multi-task streaming audio understanding: Speaker Diarization, Verification, Emotion, and Noise-Invariant Sensing.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cached analyzer instance
_analyzer: Optional[UnifiedAudioAnalyzer] = None


def get_analyzer() -> UnifiedAudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        # Load analyzer in non-strict demo fallback mode if missing
        _analyzer = UnifiedAudioAnalyzer(strict_checkpoints=False, demo_mode=True)
    return _analyzer


class HealthResponse(BaseModel):
    status: str
    hardware_device: str
    cuda_available: bool
    checkpoints: Dict[str, bool]
    onnx_engines: Dict[str, bool]
    version: str


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    """Verify system health, GPU hardware acceleration, and model provenance."""
    dev = get_device(verbose=False)
    ana = get_analyzer()
    
    onnx_audio = (CHECKPOINTS_DIR / "se_audiocnn.onnx").exists()
    onnx_spk = (CHECKPOINTS_DIR / "se_speaker_embedding.onnx").exists()
    
    return HealthResponse(
        status="healthy",
        hardware_device=str(dev),
        cuda_available=torch.cuda.is_available(),
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


@app.post("/api/v1/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    num_speakers: int = Form(0),
    apply_noise: bool = Form(False),
    noise_class: Optional[str] = Form(None),
    snr_db: float = Form(10.0),
):
    """Analyze uploaded audio file using the unified multi-task intelligence pipeline."""
    try:
        content = await file.read()
        audio_io = io.BytesIO(content)
        waveform, sr = sf.read(audio_io, dtype="float32")
        
        # Convert multi-channel to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1 if waveform.shape[1] < waveform.shape[0] else 0)

        # Resample if sr != SAMPLE_RATE
        if sr != SAMPLE_RATE:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

        if apply_noise and noise_class:
            from pipeline.urban_noise_mixer import UrbanNoiseMixer
            mixer = UrbanNoiseMixer()
            waveform, _ = mixer.mix_noise(waveform, noise_class=noise_class, target_snr=snr_db)
            
        ana = get_analyzer()
        report = ana.analyze_waveform(
            waveform,
            file_name=file.filename or "uploaded_audio.wav",
            num_speakers=num_speakers if num_speakers > 0 else 2,
        )
        report["original_filename"] = file.filename
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding or analysis error: {str(e)}")


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
        "timestamp_start": 0.0,
        "timestamp_end": 0.5,
        "speaker_tag": "Speaker_1",
        "confidence": 0.94,
        "is_speech": true
      }
    """
    await websocket.accept()
    
    diarizer = StreamingDiarizer(
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
