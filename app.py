"""Interactive Streamlit Dashboard for UrbanSound8K Speech & Audio Intelligence.
Integrates:
  - Environmental Sound Recognition (UrbanSound8K)
  - Real-World Urban Noise Injection (SNR controls)
  - Speaker Diarization ('Who spoke when?')
  - Speaker Identification (VoxCeleb voiceprints)
  - Speech Emotion Recognition (RAVDESS 8 emotions)
"""
import io
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from pipeline.config import BASE_DIR, AUDIO_DIR, SAMPLE_RATE, CLASSES
from pipeline.audio_loader import load_and_preprocess_audio, compute_log_mel_spectrogram
from pipeline.urban_noise_mixer import UrbanNoiseMixer
from pipeline.unified_analyzer import UnifiedAudioAnalyzer

st.set_page_config(
    page_title="UrbanSound Audio Intelligence Hub",
    page_icon="🎙️",
    layout="wide",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1E88E5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎙️ UrbanSound8K Speech & Audio Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Speaker Diarization • Speaker Identification • Emotion Recognition • Noise-Invariant Acoustic Sensing</div>', unsafe_allow_html=True)

# Cache analyzer in session state
@st.cache_resource
def get_analyzer():
    analyzer = UnifiedAudioAnalyzer()
    vox_root = BASE_DIR / "wav"
    for spk_id in ["id10001", "id10002", "id10003"]:
        spk_dir = vox_root / spk_id
        if spk_dir.exists():
            wavs = list(spk_dir.glob("*/*.wav"))
            if wavs:
                analyzer.enroll_known_speaker(f"Speaker_{spk_id}", wavs[0])
    return analyzer

analyzer = get_analyzer()

# Sidebar: Hardware Engine Info
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    st.sidebar.success(f"🟢 **Hardware:** {gpu_name} ({vram:.1f} GB)")
else:
    st.sidebar.warning("⚠️ **Hardware:** CPU Mode")

# Sidebar: Input Selection
st.sidebar.header("📁 Audio Input Source")
input_mode = st.sidebar.radio("Choose Input Mode", ["Curated Dataset Samples", "Upload Custom WAV"])

selected_file = None

if input_mode == "Curated Dataset Samples":
    category = st.sidebar.selectbox(
        "Dataset Category",
        ["RAVDESS Speech Emotion (wo/)", "UrbanSound8K Background Sound", "VoxCeleb Speaker (wav/)"]
    )
    if "RAVDESS" in category:
        root_dir = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24"
        files = list(root_dir.glob("Actor_*/*.wav"))[:25]
    elif "UrbanSound" in category:
        root_dir = AUDIO_DIR
        files = list(root_dir.glob("fold1/*.wav"))[:25]
    else:
        root_dir = BASE_DIR / "wav"
        files = list(root_dir.glob("id10001/*/*.wav"))[:25]

    file_names = [f"{f.parent.name}/{f.name}" for f in files]
    idx = st.sidebar.selectbox("Select File", range(len(file_names)), format_func=lambda i: file_names[i])
    if files:
        selected_file = files[idx]
else:
    uploaded = st.sidebar.file_uploader("Upload .wav file", type=["wav"])
    if uploaded is not None:
        temp_path = BASE_DIR / "pipeline_output" / "temp_upload.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        selected_file = temp_path

# Noise Injection Settings
st.sidebar.markdown("---")
st.sidebar.header("🔊 UrbanSound8K Noise Injection")
inject_noise = st.sidebar.checkbox("Simulate Real-World Urban Noise", value=False)
noise_class = None
target_snr = 10.0

if inject_noise:
    noise_class = st.sidebar.selectbox("Background Noise Event", CLASSES, index=8) # Default: siren
    target_snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", min_value=-5.0, max_value=25.0, value=10.0, step=1.0)
    st.sidebar.caption("Lower SNR = louder background street noise (0 dB = equal volume)")

num_speakers = st.sidebar.number_input("Max Diarization Speakers", min_value=1, max_value=5, value=2)

# Main Dashboard Execution
if selected_file and selected_file.exists():
    waveform = load_and_preprocess_audio(selected_file)
    applied_noise_name = "None (Clean Audio)"
    current_snr = None

    if inject_noise:
        mixer = UrbanNoiseMixer(classes=[noise_class])
        waveform, applied_noise_name, current_snr = mixer.mix(waveform, snr_db=target_snr)

    # 1. Audio Player & Visualizer
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎧 Audio Playback")
        buf = io.BytesIO()
        sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")
        st.caption(f"Source: `{selected_file.name}` | Noise: **{applied_noise_name}**" + (f" @ {current_snr:.1f} dB" if current_snr is not None else ""))

    with col2:
        st.subheader("📊 Audio Waveform & Spectrogram")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.2), sharex=False)
        time_axis = np.linspace(0, len(waveform) / SAMPLE_RATE, len(waveform))
        ax1.plot(time_axis, waveform, color="#1E88E5", lw=0.8)
        ax1.set_ylabel("Amp")
        ax1.set_xlim(0, len(waveform) / SAMPLE_RATE)
        ax1.grid(True, alpha=0.3)

        spec = compute_log_mel_spectrogram(waveform)
        im = ax2.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Freq")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # 2. Multi-Task Unified Analysis
    if st.button("🚀 Run Multi-Task Audio Intelligence Pipeline", type="primary"):
        with st.spinner("Analyzing audio with UrbanSound8K, Diarization, Speaker ID & Emotion networks..."):
            # Save current processed waveform to temp file for analyzer
            temp_eval = BASE_DIR / "pipeline_output" / "eval_sample.wav"
            temp_eval.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(temp_eval), waveform, SAMPLE_RATE)

            report = analyzer.analyze_audio_file(temp_eval, num_speakers=num_speakers)

            # Overview Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Duration", report["duration"])
            with m2:
                st.metric("Detected Noise Scene", report["environmental_acoustic_scene"]["detected_class"].replace("_", " ").title())
            with m3:
                st.metric("Scene Confidence", f"{report['environmental_acoustic_scene']['confidence']*100:.1f}%")
            with m4:
                st.metric("Speech Turns", report["total_speech_turns"])

            st.markdown("### 🗣️ Speaker Diarization & Emotion Recognition Timeline")
            turns = report["speaker_diarization_and_emotions"]
            if turns:
                for idx, t in enumerate(turns):
                    with st.container():
                        c_turn, c_spk, c_emo = st.columns([1, 2, 2])
                        with c_turn:
                            st.markdown(f"**Turn #{idx+1}**<br>`{t['time_range']}`", unsafe_allow_html=True)
                        with c_spk:
                            st.markdown(
                                f"👤 **Cluster:** `{t['cluster_speaker']}`<br>"
                                f"🆔 **Identified:** **{t['identified_speaker']}** ({t['speaker_confidence']*100:.1f}%)",
                                unsafe_allow_html=True,
                            )
                        with c_emo:
                            emo_color = {
                                "happy": "🟢", "calm": "🔵", "neutral": "⚪",
                                "sad": "🔵", "angry": "🔴", "fearful": "🟠",
                                "disgust": "🟤", "surprised": "🟡"
                            }.get(t["emotion"], "🟣")
                            st.markdown(
                                f"🎭 **Emotion:** {emo_color} **{t['emotion'].title()}**<br>"
                                f"📊 **Confidence:** {t['emotion_confidence']*100:.1f}%",
                                unsafe_allow_html=True,
                            )
                        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
            else:
                st.info("No continuous speech segments detected above the VAD energy threshold (environmental audio dominant).")

            with st.expander("📄 View Full JSON Ruflo Session Report"):
                st.json(report)
else:
    st.info("Please select or upload an audio file from the left sidebar to begin.")
