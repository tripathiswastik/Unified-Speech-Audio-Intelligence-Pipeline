"""Single Audio File Inference & Top-k Prediction for UrbanSound8K."""
import argparse
from pathlib import Path
import torch
import numpy as np

from pipeline.config import CHECKPOINTS_DIR, CLASSES, ID_TO_CLASS
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.model import AudioCNN


def predict_audio(audio_path: str, checkpoint_path: str = None, top_k: int = 3):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if checkpoint_path is None:
        checkpoints = list(CHECKPOINTS_DIR.glob("best_model_fold*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No trained checkpoints found in pipeline_output/checkpoints/")
        checkpoint_path = checkpoints[0]
    else:
        checkpoint_path = Path(checkpoint_path)

    from pipeline.device import get_device
    device = get_device()

    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AudioCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    waveform = load_and_preprocess_audio(audio_path)
    spec = compute_log_mel_spectrogram(waveform)
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(spec_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [
        {"class": ID_TO_CLASS[idx], "probability": float(probs[idx])}
        for idx in top_indices
    ]

    print(f"\nPredictions for {audio_path.name}:")
    for r in results:
        bar = "█" * int(r["probability"] * 30)
        print(f"  {r['class']:<20}: {r['probability'] * 100:6.2f}% | {bar}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sound class for a wav file")
    parser.add_argument("audio", type=str, help="Path to audio .wav file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    predict_audio(args.audio, args.checkpoint, top_k=args.top_k)
