"""Live Full-Duplex WebSocket Client for UrbanSound Streaming Gateway.

Can be run directly via CLI:
    python -m pipeline.live_client --url ws://127.0.0.1:8000/ws/stream --chunks 10
"""
import sys
import time
import json
import argparse
import threading
from typing import List, Dict, Any
import numpy as np
from websockets.sync.client import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def run_live_client(ws_url: str = "ws://127.0.0.1:8000/ws/stream", num_chunks: int = 10):
    """Run full-duplex live stream verification against running FastAPI server."""
    print("=" * 75)
    print(f" [CLIENT] FULL-DUPLEX WEBSOCKET STREAMING TEST: {ws_url}")
    print("=" * 75, flush=True)

    received_messages: List[Dict[str, Any]] = []
    stop_receiver = threading.Event()

    def receiver_loop(ws):
        while not stop_receiver.is_set():
            try:
                msg = ws.recv()
                if msg:
                    data = json.loads(msg)
                    received_messages.append(data)
                    event_type = data.get("event", "unknown")
                    print(f"  <-- [SERVER EVENT] {event_type.upper()}: {data}", flush=True)
            except Exception:
                break

    print(f"\n[Step 1] Connecting to {ws_url}...", flush=True)
    t0 = time.perf_counter()
    with connect(ws_url, close_timeout=3.0) as ws:
        rcv_thread = threading.Thread(target=receiver_loop, args=(ws,), daemon=True)
        rcv_thread.start()

        time.sleep(0.3)
        assert len(received_messages) > 0, "No greeting message received from server!"
        greeting = received_messages[0]
        print(f"  [CONNECTED] Protocol handshake in {(time.perf_counter()-t0)*1000:.1f}ms: {greeting}\n")

        # Negotiate PCM16 encoding
        ws.send(json.dumps({"action": "configure", "encoding": "pcm16"}))
        time.sleep(0.2)

        sample_rate = greeting.get("sample_rate", 22050)
        chunk_samples = int(sample_rate * 0.25)  # 250ms chunks

        print(f"[Step 2] Streaming {num_chunks} audio chunks ({chunk_samples} samples, 250ms each)...", flush=True)
        for i in range(num_chunks):
            freq = 300.0 + (i % 2) * 200.0
            t = np.linspace(0, 0.25, chunk_samples, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            pcm16_bytes = (audio * 32767).astype(np.int16).tobytes()

            t_send = time.perf_counter()
            ws.send(pcm16_bytes)
            dt_send = (time.perf_counter() - t_send) * 1000.0
            print(f"  --> Sent chunk #{i+1:02d} ({len(pcm16_bytes)} bytes) in {dt_send:.2f}ms", flush=True)
            time.sleep(0.08)

        # Allow buffer processing
        time.sleep(0.6)

        # Step 3: Send session reset
        print("\n[Step 3] Testing session reset command...", flush=True)
        ws.send(json.dumps({"action": "reset"}))
        time.sleep(0.3)

        stop_receiver.set()

    print("\n" + "=" * 75)
    print(" [SUMMARY] STREAMING VERIFICATION COMPLETE")
    print(f"  * Total Server Events Received: {len(received_messages)}")
    events_by_type = {}
    for m in received_messages:
        ev = m.get("event", "unknown")
        events_by_type[ev] = events_by_type.get(ev, 0) + 1
    for ev, count in events_by_type.items():
        print(f"    - {ev}: {count}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Live WebSocket streaming client")
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8000/ws/stream", help="WebSocket server URL")
    parser.add_argument("--chunks", type=int, default=10, help="Number of chunks to stream")
    args = parser.parse_args()

    run_live_client(ws_url=args.url, num_chunks=args.chunks)


if __name__ == "__main__":
    main()
