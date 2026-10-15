# Unified Speech & Audio Intelligence Pipeline

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.23+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end multi-task audio understanding pipeline combining **Speaker Diarization**, **Speaker Identification**, and **Speech Emotion Recognition (SER)** fortified by **UrbanSound8K** realistic environmental noise invariance.

---

## 🌟 Key Features

1. **Environmental Acoustic Sensing (UrbanSound8K)**:
   - Identifies background acoustic scenes across 10 classes (`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`).
   - Deep 4-block 2D-CNN with Global Average Pooling on 128-band Log-Mel Spectrograms.

2. **Real-World Urban Noise Invariance (`UrbanNoiseMixer`)**:
   - Injects realistic background noise into speech at variable Signal-to-Noise Ratios (SNR from -5 dB to 25 dB).
   - Guarantees models remain resilient in real-world deployment (street traffic, drilling, sirens).

3. **Speaker Identification (VoxCeleb Voiceprints)**:
   - Deep speaker embedding network with Temporal Statistics Pooling producing 128-d L2-normalized voiceprints.
   - Enrolls custom voices and verifies speakers via cosine similarity.

4. **Speaker Diarization ("Who Spoke When")**:
   - Voice Activity Detection (VAD) coupled with sliding-window Agglomerative Clustering on cosine distances to partition multi-speaker dialogue into coherent temporal speaker turns.

5. **Speech Emotion Recognition (RAVDESS SER)**:
   - 8-class emotional classification (*neutral, calm, happy, sad, angry, fearful, disgust, surprised*).
   - Speaker-independent evaluation splits (trained on actors 1–20, validated on unseen test actors 21–24).

6. **Interactive Streamlit Web Dashboard**:
   - Live audio playback, dual waveform & Log-Mel spectrogram visualization.
   - Real-time environmental noise simulation slider with live playback.
   - Multi-speaker timeline rendering with emotion gauges and confidence scores.

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
├── requirements.txt           # Project dependencies
├── pipeline/
│   ├── config.py              # Central hyperparameters & paths
│   ├── audio_loader.py        # Robust loader (24-bit PCM / extensible wav, Log-Mel)
│   ├── urban_noise_mixer.py   # SNR-scaled realistic noise injection
│   ├── dataset.py             # UrbanSound8K PyTorch Dataset with fold isolation
│   ├── model.py               # AudioCNN 2D-ConvNet architecture
│   ├── emotion_dataset.py     # RAVDESS Dataset with speaker-independent splits
│   ├── emotion_model.py       # EmotionClassifierCNN (8 emotion classes)
│   ├── speaker_id.py          # SpeakerEmbeddingNet & SpeakerIdentifier registry
│   ├── diarization.py         # SpeakerDiarizer (VAD + Agglomerative Clustering)
│   ├── unified_analyzer.py    # Master multi-task analyzer pipeline
│   ├── train.py               # Environmental sound classifier trainer
│   ├── train_emotion.py       # SER trainer with UrbanSound8K noise injection
│   ├── inference.py           # Single-file CLI predictor
│   ├── ruflo_tracker.py       # Ruflo experiment lineage & evaluation recorder
│   └── ravdess_metadata.csv   # Indexed RAVDESS speech dataset manifest
└── .gitignore                 # Excludes raw audio data archives & weights
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/tripathiswastik/Unified-Speech-Audio-Intelligence-Pipeline.git
cd Unified-Speech-Audio-Intelligence-Pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Visit `http://localhost:8501` to test with curated audio samples or upload custom audio.

### 3. Run Multi-Task Analysis from CLI
```bash
python -m pipeline.unified_analyzer
```

### 4. Train Models

**Train UrbanSound8K Environmental Classifier (Hold out Fold 1):**
```bash
python -m pipeline.train --fold 1 --epochs 10 --batch-size 32
```

**Train Speech Emotion Recognition with Urban Noise Augmentation:**
```bash
python -m pipeline.train_emotion --epochs 10 --batch-size 32
```

---

## 📊 Evaluation & Benchmarking

The pipeline uses strict validation protocols:
- **UrbanSound8K**: Strict 10-fold cross validation preserving official recording groupings (prevents recording ID leakage).
- **RAVDESS**: Speaker-independent holdout (Actors 1–20 for training, Actors 21–24 for testing).
- **Metrics**: Accuracy, Macro-F1, and per-class confusion matrices tracked via `RufloExperimentTracker`.

---

## 📜 License
MIT License. See LICENSE for details.
