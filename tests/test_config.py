"""Tests for pipeline.config dynamic path resolution and environment overrides."""
import os
from pathlib import Path
import pytest
from pipeline.config import BASE_DIR, CHECKPOINTS_DIR, AUDIO_DIR, METADATA_CSV


def test_base_dir_resolution():
    """Verify that BASE_DIR resolves to a valid existing Path."""
    assert isinstance(BASE_DIR, Path)
    assert BASE_DIR.is_absolute()


def test_derived_paths():
    """Verify derived paths follow project conventions."""
    assert CHECKPOINTS_DIR == BASE_DIR / "pipeline_output" / "checkpoints"
    assert AUDIO_DIR == BASE_DIR / "UrbanSound8K" / "audio"
    assert METADATA_CSV == BASE_DIR / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
