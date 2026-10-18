"""Real-Time Audio Streaming Contract & Decoding Specifications.

Provides standardized audio decoders, encoding constants, session state representations,
and event payload serializations shared across the FastAPI WebSocket server and dashboard.
"""
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np

ENCODING_FLOAT32 = "float32"
ENCODING_PCM16 = "pcm16"
SUPPORTED_ENCODINGS = (ENCODING_FLOAT32, ENCODING_PCM16)

DEFAULT_SAMPLE_RATE = 16000
MAX_SAMPLE_RATE = 48000
MIN_SAMPLE_RATE = 8000


def decode_pcm(raw_bytes: bytes, encoding: str = ENCODING_FLOAT32) -> np.ndarray:
    """Decode raw bytes into a 1D float32 NumPy array normalized to [-1.0, 1.0].
    
    Args:
        raw_bytes: Binary audio payload.
        encoding: "float32" (default) or "pcm16".
        
    Returns:
        1D float32 NumPy array.
        
    Raises:
        ValueError: If byte length is not aligned to element size,
                    encoding is unsupported, or samples contain NaN/Inf.
    """
    if not raw_bytes:
        return np.array([], dtype=np.float32)

    enc = encoding.lower().strip()
    if enc not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported audio encoding '{encoding}'. Supported: {SUPPORTED_ENCODINGS}")

    if enc == ENCODING_FLOAT32:
        if len(raw_bytes) % 4 != 0:
            raise ValueError(
                f"Ragged byte length {len(raw_bytes)} for float32 PCM: byte count must be a multiple of 4."
            )
        samples = np.frombuffer(raw_bytes, dtype=np.float32)
    elif enc == ENCODING_PCM16:
        if len(raw_bytes) % 2 != 0:
            raise ValueError(
                f"Ragged byte length {len(raw_bytes)} for pcm16: byte count must be a multiple of 2."
            )
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        samples = int16_arr.astype(np.float32) / 32768.0

    if len(samples) > 0 and not np.all(np.isfinite(samples)):
        raise ValueError("Decoded audio chunk contains non-finite values (NaN or Inf).")

    return samples


class ChunkQueue:
    """Bounded FIFO sliding ring buffer for sequential audio chunk ingestion."""

    def __init__(self, max_chunks: int = 100):
        self._queue: deque = deque(maxlen=max_chunks)

    def push(self, chunk: np.ndarray) -> None:
        if len(chunk) > 0:
            self._queue.append(chunk)

    def pop(self) -> Optional[np.ndarray]:
        if self._queue:
            return self._queue.popleft()
        return None

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)


def event_payload(ev: Any, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Serialize a StreamingSpeechEvent into a standardized WebSocket event payload."""
    payload = {
        "event": "speech_segment",
        "event_type": getattr(ev, "event_type", "window"),
        "timestamp_start": round(float(getattr(ev, "timestamp_start", 0.0)), 3),
        "timestamp_end": round(float(getattr(ev, "timestamp_end", 0.0)), 3),
        "speaker_tag": str(getattr(ev, "speaker_tag", "Unknown")),
        "confidence": round(float(getattr(ev, "confidence", 0.0)), 3),
        "is_speech": bool(getattr(ev, "is_speech", False)),
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def error_payload(code: str, message: str, fatal: bool = False) -> Dict[str, Any]:
    """Construct a structured WebSocket error event."""
    return {
        "event": "error",
        "code": code,
        "message": message,
        "fatal": fatal,
    }
