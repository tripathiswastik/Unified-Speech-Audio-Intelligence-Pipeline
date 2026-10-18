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


def safe_torch_load(path, map_location=None, weights_only: bool = False):
    """Cross-version safe torch.load helper.

    In PyTorch 2.6+, weights_only defaults to True, raising _pickle.UnpicklingError
    when loading checkpoints containing NumPy RNG states, metadata dictionaries,
    or composite artifacts. This helper safely passes weights_only=False where
    supported and falls back gracefully for older PyTorch versions (< 2.4).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

