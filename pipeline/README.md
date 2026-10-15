# UrbanSound8K Unified Speech & Audio Intelligence Pipeline

This pipeline implements a multi-task audio understanding system combining **Urban Environmental Sound Classification**, **Speaker Diarization**, **Speaker Identification**, and **Speech Emotion Recognition (SER)**.

---

## 1. Unified Dataset Architecture

| Task | Primary Dataset | Role & Integration |
| :--- | :--- | :--- |
| **Environmental Acoustic Sensing** | `UrbanSound8K` (8,732 `.wav` files, 10 classes) | Detects background acoustic scenes (drilling, sirens, car horns, street music). |
| **Real-World Noise Invariance** | `UrbanSound8K` + `UrbanNoiseMixer` | Blends dynamic environmental noises into speech at configurable SNR (-5 dB to 20 dB) to make models robust in noisy urban conditions. |
| **Speaker Identification** | `wav/` (VoxCeleb, 410 speaker folders) | Provides multi-speaker voiceprints for training embedding networks and enrollment. |
| **Speaker Diarization** | `wav/` + `wo/` | Performs Voice Activity Detection (VAD) + sliding-window Agglomerative Clustering to answer **"Who spoke when?"** |
| **Speech Emotion Recognition** | `wo/Audio_Speech_Actors_01-24` (RAVDESS, 1,440 files) | Classifies 8 emotional states (*neutral, calm, happy, sad, angry, fearful, disgust, surprised*) on speaker-isolated turns. |

---

## 2. Directory Layout

```
d:/ag-ai/UrbanSound/pipeline/
├── config.py                 # Central configurations, sample rates, classes
├── audio_loader.py           # Robust loader (24-bit PCM / extensible wav, mono, 22.05kHz, Log-Mel)
├── urban_noise_mixer.py      # Dynamic SNR-scaled background noise blender
├── dataset.py                # UrbanSound8K PyTorch Dataset with fold isolation & SpecAugment
├── model.py                  # AudioCNN 2D-ConvNet for environmental sound classification
├── emotion_dataset.py        # RAVDESS Dataset with speaker-independent splits & noise augmentation
├── emotion_model.py          # EmotionClassifierCNN (8 emotion classes)
├── speaker_id.py             # SpeakerEmbeddingNet (128-d voiceprints) & SpeakerIdentifier registry
├── diarization.py            # SpeakerDiarizer (VAD + Agglomerative Clustering)
├── unified_analyzer.py       # End-to-end multi-task analyzer combining all models
├── train.py                  # UrbanSound8K training runner
├── train_emotion.py          # SER training runner with UrbanSound8K noise injection
├── inference.py              # Single audio file predictor
└── ruflo_tracker.py          # Ruflo experiment lineage and evaluation metrics recorder
```

---

## 3. How to Run

Activate the virtual environment first:
```powershell
. d:\ag-ai\.venv\Scripts\Activate.ps1
```

### 1. Train UrbanSound8K Environmental Classifier (e.g., Fold 1)
```powershell
python -m pipeline.train --fold 1 --epochs 10 --batch-size 32
```

### 2. Train Speech Emotion Recognition with UrbanSound8K Noise Augmentation
```powershell
python -m pipeline.train_emotion --epochs 10 --batch-size 32
```

### 3. Run End-to-End Unified Analysis (Diarization + Speaker ID + Emotion + Background Noise)
```powershell
python -m pipeline.unified_analyzer
```
