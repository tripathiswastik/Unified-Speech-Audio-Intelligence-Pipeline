"""Stand-alone ONNX export utility for CI and edge deployment."""
from pathlib import Path
from pipeline.onnx_exporter import run_edge_preparation

if __name__ == "__main__":
    run_edge_preparation()
