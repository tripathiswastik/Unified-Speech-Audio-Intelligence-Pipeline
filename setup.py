from setuptools import setup, find_packages

setup(
    name="unified-speech-pipeline",
    version="1.0.0",
    description="Unified Speech & Audio Intelligence Pipeline with Environmental Noise Invariance",
    author="Swastik Tripathi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "torch>=1.13.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "streamlit>=1.22.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
)
