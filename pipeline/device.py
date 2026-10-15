"""Centralized Hardware & GPU Device Configuration.
Prioritizes NVIDIA GPU (CUDA) for high-performance training, testing, and inference.
"""
import torch


def get_device(verbose: bool = True) -> torch.device:
    """Return torch.device('cuda:0') if NVIDIA GPU is available, else cpu."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Hardware Engine] Using NVIDIA GPU: {gpu_name} ({vram_gb:.2f} GB VRAM) - CUDA Enabled")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Hardware Engine] CUDA not detected. Falling back to CPU.")
    return device
