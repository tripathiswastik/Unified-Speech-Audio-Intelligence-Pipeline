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
        res = run_edge_preparation()
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
        except Exception:
            pass

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
            import pkg_resources
            print("\nInstalled Packages:", file=sys.stderr)
            for p in sorted(pkg_resources.working_set, key=lambda p: p.project_name.lower()):
                print(f"  {p.project_name}=={p.version}", file=sys.stderr)
        except Exception:
            pass

        sys.stderr.flush()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()
