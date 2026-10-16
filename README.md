# Unified Speech & Audio Intelligence Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![CI](https://github.com/tripathiswastik/Unified-Speech-Audio-Intelligence-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/tripathiswastik/Unified-Speech-Audio-Intelligence-Pipeline/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end multi-task audio understanding pipeline combining **Speaker Diarization**, **Speaker Identification**, and **Speech Emotion Recognition (SER)** fortified by **UrbanSound8K** realistic environmental noise invariance.

---

## 🌟 Key Features

1. **Environmental Acoustic Sensing (UrbanSound8K)**:
   - Identifies background acoustic scenes across 10 classes (`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`).
   - Deep 4-block 2D-CNN with Global Average Pooling on 128-band Log-Mel Spectrograms.

2. **Real-World Urban Noise Augmentation (`UrbanNoiseMixer`)**:
   - Injects realistic background noise into speech at variable Signal-to-Noise Ratios (SNR from -5 dB to 25 dB).
   - Enhances acoustic robustness in real-world deployment (street traffic, drilling, sirens) with seamless multi-clip crossfading for long audio streams.

3. **High-Throughput Speaker Identification (VoxCeleb Voiceprints)**:
   - Deep speaker embedding network with Temporal Statistics Pooling producing 128-d L2-normalized voiceprints.
   - Vectorized batch matrix multiplication (`np.dot`) against enrolled voiceprint registries.

4. **Batched Speaker Diarization with Whole-File STFT Slicing**:
   - Voice Activity Detection (VAD) coupled with whole-file Mel-spectrogram slicing (eliminating redundant STFT computations across sliding windows).
   - Enforces strictly non-overlapping chronological speaker turn boundaries ($turn_i.end \le turn_{i+1}.start$) and supports automatic speaker count estimation ($k$ via silhouette scoring).

5. **Speech Emotion Recognition (RAVDESS SER)**:
   - 8-class emotional classification (*neutral, calm, happy, sad, angry, fearful, disgust, surprised*).
   - Strict 3-way speaker-independent disjoint splits: Train (Actors 1–18, 75%), Validation (Actors 19–20, 8.3% for checkpoint selection), and Test (Actors 21–24, 16.7% held-out for unbiased benchmarking).

6. **Interactive Streamlit Web Dashboard**:
   - Live audio playback, dual waveform & Log-Mel spectrogram visualization.
   - Real-time environmental noise simulation slider with live playback.
   - Multi-speaker timeline rendering with emotion gauges, confidence scores, and checkpoint provenance badges.

---

## ⚡ Performance & Engineering Optimizations

- **Cross-Platform Dynamic Path Resolution**: No hard-coded absolute paths. Paths resolve dynamically relative to project root or can be overridden via `URBANSOUND_BASE_DIR`.
- **Whole-File STFT & Mel Slicing**: Computes STFT once per continuous waveform and extracts frame slices $O(1)$ for sliding windows, cutting diarization feature extraction latency by up to 10×.
- **Zero-Disk In-Memory Waveform Analysis**: Multi-task inference accepts in-memory NumPy waveforms directly, eliminating temporary disk I/O and race conditions.
- **Precomputed Mel Filterbank Reuse**: Mel triangular basis matrices are allocated once per configuration, computing mel power via direct linear algebra `np.dot(mel_basis, |STFT|^2)` rather than rebuilding filterbanks per sample.
- **Dual-Tier Bounded LRU & Disk Caching**: In-memory spectrogram cache is capped with an LRU policy (`maxsize=2000`) and disk persistence to prevent memory bloat or out-of-memory crashes on large corpora.
- **Streaming Batch Inference & Memory Capping**: Window embeddings stream in chunks of `batch_size=32`, avoiding simultaneous allocation of all window spectrograms in RAM.
- **Strict Timeline Invariants**: Merges contiguous turns of the same speaker and resolves inter-speaker transition boundaries to midpoints, guaranteeing no accidental overlapping turns.
- **Untrained Model Safeguards**: The analyzer flags and surfaces checkpoint provenance (`TRAINED_CHECKPOINT` vs `RANDOMLY_INITIALIZED_UNTRAINED`) and enforces fail-closed checks by default (`strict_checkpoints=True`).

---

## 🏗️ Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 UrbanSound8K (8,732 audio files)       │
                  │   10 acoustic scenes: siren, drilling, horn, etc.     │
                  └───────────────────────────┬────────────────────────────┘
                                              │ (Realistic Noise Injection &
                                              │  Acoustic Context Detection)
                                              ▼
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
│   wav/ (VoxCeleb)       │       │ Continuous Audio Stream │       │   wo/ (RAVDESS Speech)  │
│ 410 Speaker Directories │       │  (Multi-Speaker Convo)  │       │ 1,440 Speech Utterances │
└────────────┬────────────┘       └───────────┬─────────────┘       └────────────┬────────────┘
             │                                │                                  │
             ▼                                ▼                                  ▼
   [Speaker Embeddings]            [Speaker Diarizer]                 [Emotion Classifier]
    "Who is speaking?"              "Who spoke when?"                 "How do they feel?"
     (128-d d-vector)               (VAD + Clustering)                 (8-class SER)
             │                                │                                  │
             └────────────────────────────────┼──────────────────────────────────┘
                                              ▼
                             ┌──────────────────────────────────┐
                             │ Unified Audio Intelligence Report│
                             │   (Logged to Ruflo Tracker)      │
                             └──────────────────────────────────┘
```

---

## 📂 Project Structure

```
.
├── app.py                     # Interactive Streamlit Web Dashboard
├── requirements.txt           # Project dependencies & testing tools
├── tests/                     # Automated unit and integration test suite
│   ├── test_config.py
│   ├── test_audio_loader.py
│   ├── test_speaker_and_diarization.py
│   └── test_inference_pipeline.py
├── .github/workflows/
│   └── ci.yml                 # Cross-platform GitHub Actions CI (Ubuntu & Windows, Py 3.10/3.11)
├── pipeline/
│   ├── config.py              # Central hyperparameters & dynamic paths
│   ├── validate_environment.py# Preflight sanity checker for data & weights
│   ├── audio_loader.py        # Robust loader with precomputed Mel filterbanks
│   ├── urban_noise_mixer.py   # SNR-scaled realistic noise injection
│   ├── dataset.py             # UrbanSound8K PyTorch Dataset with bounded LRU cache
│   ├── model.py               # AudioCNN 2D-ConvNet architecture
│   ├── emotion_dataset.py     # RAVDESS Dataset with speaker-independent splits
│   ├── emotion_model.py       # EmotionClassifierCNN (8 emotion classes)
│   ├── speaker_id.py          # SpeakerEmbeddingNet & Vectorized SpeakerIdentifier
│   ├── diarization.py         # Batched SpeakerDiarizer (VAD + Agglomerative Clustering)
│   ├── unified_analyzer.py    # Master multi-task analyzer pipeline with provenance guards
│   ├── train.py               # Environmental sound classifier trainer
│   ├── train_emotion.py       # SER trainer with UrbanSound8K noise injection
│   ├── train_speaker_id.py    # Speaker embedding network trainer (VoxCeleb)
│   ├── inference.py           # Single-file CLI predictor
│   ├── ruflo_tracker.py       # Ruflo experiment lineage & evaluation recorder
│   └── ravdess_metadata.csv   # Indexed RAVDESS speech dataset manifest
└── models/                    # Checkpoints (.pt) and speaker enrollment databases
```

---

## 📁 Dataset Directory Layout

To train or run full evaluations with the raw datasets, organize files according to the layout below:

```
UrbanSound/
├── UrbanSound8K.csv           # UrbanSound8K metadata table
├── fold1/ ... fold10/         # Audio clips partitioned into 10 folds (.wav)
├── wo/                        # RAVDESS Audio_Speech_Actors_01-24 (.wav)
│   ├── Actor_01/
│   └── ...
└── wav/                       # VoxCeleb speaker audio directories
    ├── id10001/
    └── ...
```

*(Note: The repository includes synthetic demonstration fallbacks and pre-indexed metadata manifests for evaluation even if full raw gigabyte audio archives are absent).*

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and set up a Python 3.10+ virtual environment:
```bash
git clone https://github.com/tripathiswastik/Unified-Speech-Audio-Intelligence-Pipeline.git
cd Unified-Speech-Audio-Intelligence-Pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Preflight Audit
Verify your environment, dependencies, audio datasets, and trained model weights:
```bash
python -m pipeline.validate_environment
```

### 3. Run Automated Tests
Execute the comprehensive test suite across loaders, diarization, speaker ID, and inference pipelines:
```bash
pytest tests/ -v
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Visit `http://localhost:8501` to test with curated audio samples or upload custom audio.

### 5. Run Multi-Task Analysis from CLI
```bash
python -m pipeline.unified_analyzer
```

### 6. Training Pipelines

**Train UrbanSound8K Environmental Classifier (Hold out Fold 1):**
```bash
python -m pipeline.train --fold 1 --epochs 10 --batch-size 32
```

**Train Speech Emotion Recognition with Urban Noise Augmentation:**
```bash
python -m pipeline.train_emotion --epochs 10 --batch-size 32
```

**Train VoxCeleb Speaker Embedding Network:**
```bash
python -m pipeline.train_speaker_id --epochs 15 --batch-size 32
```

---

## 📊 Evaluation & Benchmarking

The pipeline uses strict validation protocols:
- **UrbanSound8K**: Strict 10-fold cross validation preserving official recording groupings (prevents recording ID leakage).
- **RAVDESS**: Speaker-independent holdout (Actors 1–20 for training, Actors 21–24 for testing).
- **Speaker ID**: Cosine verification against VoxCeleb voiceprint benchmarks.
- **Metrics**: Accuracy, Macro-F1, and per-class confusion matrices tracked via `RufloExperimentTracker`.

---

## 📜 License
MIT License. See LICENSE for details.
