"""Stand-alone ONNX export utility for CI and edge deployment."""
import os
import sys
import traceback
from pathlib import Path


def main():
    """Run edge export with detailed diagnostics and stack traces on failure."""
    print("=" * 65)
    print("Starting Edge ONNX Export Process")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 65)

    try:
        from pipeline.onnx_exporter import run_edge_preparation
        allow_untrained = os.getenv("ALLOW_UNTRAINED_ONNX", "0").strip().lower() in ("1", "true", "yes")
        res = run_edge_preparation(strict_checkpoints=not allow_untrained)
        print("\n[SUCCESS] Edge models exported successfully.")
        return res
    except Exception as exc:
        print(f"\n[FATAL ERROR during ONNX Export]: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        print("\n=== Environment Diagnostics ===", file=sys.stderr)
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<not set>')}", file=sys.stderr)
        try:
            import torch
            print(f"PyTorch Version: {torch.__version__}", file=sys.stderr)
            print(f"CUDA Available: {torch.cuda.is_available()}", file=sys.stderr)
        except ImportError as torch_err:
            print(f"PyTorch not available: {torch_err}", file=sys.stderr)

        try:
            import onnx
            print(f"ONNX Version: {getattr(onnx, '__version__', 'unknown')}", file=sys.stderr)
        except Exception as onnx_err:
            print(f"ONNX Import Error: {onnx_err}", file=sys.stderr)

        try:
            import onnxruntime
            print(f"ONNX Runtime Version: {getattr(onnxruntime, '__version__', 'unknown')}", file=sys.stderr)
        except Exception as ort_err:
            print(f"ONNX Runtime Import Error: {ort_err}", file=sys.stderr)

        try:
            import importlib
            onnxscript = importlib.import_module("onnxscript")
            print(f"ONNX Script Version: {getattr(onnxscript, '__version__', 'unknown')}", file=sys.stderr)
        except Exception as os_err:
            print(f"ONNX Script Import Error: {os_err}", file=sys.stderr)

        try:
            import pkg_resources
            ws = getattr(pkg_resources, "working_set", None)
            if ws is not None:
                print("\nInstalled Packages:", file=sys.stderr)
                for p in sorted(list(ws), key=lambda x: str(getattr(x, "project_name", "")).lower()):
                    print(f"  {getattr(p, 'project_name', 'unknown')}=={getattr(p, 'version', 'unknown')}", file=sys.stderr)
        except Exception as pkg_err:
            print(f"Package inspection unavailable: {pkg_err}", file=sys.stderr)

        sys.stderr.flush()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()
