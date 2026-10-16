"""Interactive Streamlit Dashboard for UrbanSound8K Speech & Audio Intelligence.
Integrates:
  - Continuous Environmental Sound Recognition (UrbanSound8K Multi-Window Ensemble)
  - Real-World Urban Noise Injection (SNR controls)
  - Speaker Diarization ('Who spoke when?') with VAD Hangover Smoothing
  - Speaker Identification & Verification (VoxCeleb Voiceprints)
  - Speech Emotion Recognition (RAVDESS 8 emotions)
  - Experiment Tracking & Ruflo Cross-Validation Analytics
"""
import io
import os
import re
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from pipeline.config import BASE_DIR, AUDIO_DIR, SAMPLE_RATE, CLASSES
from pipeline.audio_loader import load_full_audio, compute_log_mel_spectrogram
from pipeline.urban_noise_mixer import UrbanNoiseMixer
from pipeline.unified_analyzer import UnifiedAudioAnalyzer
from pipeline.device import get_device

st.set_page_config(
    page_title="UrbanSound Audio Intelligence Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #666; margin-bottom: 1.2rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1E88E5; }
    .verdict-match { background-color: #E8F5E9; border-left: 5px solid #2E7D32; padding: 12px 18px; border-radius: 6px; }
    .verdict-diff { background-color: #FFEBEE; border-left: 5px solid #C62828; padding: 12px 18px; border-radius: 6px; }
    .demo-banner { background-color: #FFF3E0; border-left: 5px solid #F57C00; padding: 10px 16px; border-radius: 6px; margin-bottom: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎙️ UrbanSound8K Speech & Audio Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Task Acoustic Intelligence: Speaker Diarization • Speaker Verification • Emotion Recognition • Noise-Invariant Sensing</div>', unsafe_allow_html=True)


# Sidebar Settings & Diagnostics
st.sidebar.header("⚙️ Configuration & Diagnostics")
allow_demo = st.sidebar.checkbox(
    "Synthetic Demo Mode (Unverified Weights)",
    value=False,
    help="Permits running the pipeline with randomly initialized weights for testing when trained checkpoints are unavailable."
)


# Cache base analyzer instance (immutable models)
@st.cache_resource
def get_base_analyzer(demo_mode: bool) -> UnifiedAudioAnalyzer:
    return UnifiedAudioAnalyzer(strict_checkpoints=not demo_mode, demo_mode=demo_mode)


analyzer: Optional[UnifiedAudioAnalyzer] = None
analyzer_ready = False

try:
    analyzer = get_base_analyzer(demo_mode=allow_demo)
    analyzer_ready = True
except FileNotFoundError as e:
    st.sidebar.error(f"⚠️ Checkpoint Guard Error: {e}")

# Maintain session-isolated speaker enrollments
if "session_enrolled_speakers" not in st.session_state:
    st.session_state["session_enrolled_speakers"] = {}
    vox_root = BASE_DIR / "wav"
    if vox_root.exists():
        for spk_id in ["id10001", "id10002", "id10003"]:
            spk_dir = vox_root / spk_id
            if spk_dir.exists():
                wavs = list(spk_dir.glob("*/*.wav"))
                if wavs:
                    try:
                        wave, _ = load_full_audio(wavs[0])
                        if analyzer_ready and analyzer is not None:
                            emb = analyzer.speaker_identifier.extract_embedding(wave)
                            st.session_state["session_enrolled_speakers"][f"Speaker_{spk_id}"] = emb
                    except Exception:
                        pass

# Synchronize session enrollments with analyzer's identifier
if analyzer_ready and analyzer is not None:
    analyzer.speaker_identifier.enrolled_profiles = dict(st.session_state["session_enrolled_speakers"])
    analyzer.speaker_identifier._update_enrolled_matrix()



# Sidebar: Hardware Engine Status
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    st.sidebar.success(f"🟢 **Hardware Acceleration:**\n{gpu_name} ({vram:.1f} GB VRAM)")
else:
    st.sidebar.warning("⚠️ **Hardware Mode:** CPU Fallback")

st.sidebar.markdown("---")

# Sidebar: Checkpoint Integrity & Provenance
st.sidebar.header("🧠 Model Checkpoints")
if analyzer_ready and analyzer is not None:
    if analyzer.urban_checkpoint_loaded:
        sha = f" (`{analyzer.urban_checkpoint_sha256}`)" if analyzer.urban_checkpoint_sha256 else ""
        st.sidebar.markdown(f"🟢 **UrbanSound8K:** Trained{sha}")
    else:
        st.sidebar.warning("⚠️ **UrbanSound8K:** Untrained (Random Weights)")

    if analyzer.emotion_checkpoint_loaded:
        sha = f" (`{analyzer.emotion_checkpoint_sha256}`)" if analyzer.emotion_checkpoint_sha256 else ""
        st.sidebar.markdown(f"🟢 **Emotion SER:** Trained{sha}")
    else:
        st.sidebar.warning("⚠️ **Emotion SER:** Untrained (Random Weights)")

    if analyzer.speaker_checkpoint_loaded:
        sha = f" (`{analyzer.speaker_checkpoint_sha256}`)" if analyzer.speaker_checkpoint_sha256 else ""
        st.sidebar.markdown(f"🟢 **Speaker ID:** Trained{sha}")
    else:
        st.sidebar.warning("⚠️ **Speaker ID:** Untrained (Random Weights)")
else:
    st.sidebar.error("❌ Models halted due to missing checkpoints. Enable 'Synthetic Demo Mode' above to proceed.")


st.sidebar.markdown("---")

# Main Navigation Tabs
tab_pipeline, tab_speaker_studio, tab_metrics = st.tabs([
    "🎙️ Multi-Task Audio Intelligence Pipeline",
    "🆔 Speaker Studio & Voiceprint Verification",
    "📈 Model Benchmarks & Cross-Validation",
])


# ==============================================================================
# TAB 1: MULTI-TASK AUDIO INTELLIGENCE PIPELINE
# ==============================================================================
with tab_pipeline:
    st.sidebar.header("📁 Audio Input Source")
    input_mode = st.sidebar.radio("Choose Input Mode", ["Curated Dataset Samples", "Upload Custom WAV"], key="pipeline_input_mode")

    selected_file: Optional[Path] = None
    is_temp_upload = False

    if input_mode == "Curated Dataset Samples":
        category = st.sidebar.selectbox(
            "Dataset Category",
            ["RAVDESS Speech Emotion (wo/)", "UrbanSound8K Background Sound", "VoxCeleb Speaker (wav/)"],
            key="pipeline_category",
        )
        if category is not None and "RAVDESS" in category:
            root_dir = BASE_DIR / "wo" / "Audio_Speech_Actors_01-24"
            files = list(root_dir.glob("Actor_*/*.wav"))[:25]
        elif category is not None and "UrbanSound" in category:
            root_dir = AUDIO_DIR
            files = list(root_dir.glob("fold1/*.wav"))[:25]
        else:
            root_dir = BASE_DIR / "wav"
            files = list(root_dir.glob("id10001/*/*.wav"))[:25]

        file_names = [f"{f.parent.name}/{f.name}" for f in files]
        idx = st.sidebar.selectbox("Select File", range(len(file_names)), format_func=lambda i: file_names[i], key="pipeline_file_idx")
        if files and idx is not None:
            selected_file = files[idx]
    else:
        uploaded = st.sidebar.file_uploader("Upload .wav file (Max 25MB)", type=["wav"], key="pipeline_upload")
        if uploaded is not None:
            # File size check
            if uploaded.size > 25 * 1024 * 1024:
                st.sidebar.error("Uploaded file exceeds maximum limit of 25MB.")
            else:
                # Write to unique secure temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(uploaded.getbuffer())
                temp_file.close()
                selected_file = Path(temp_file.name)
                is_temp_upload = True

    # Noise Injection Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🔊 UrbanSound8K Noise Injection")
    inject_noise = st.sidebar.checkbox("Simulate Real-World Urban Noise", value=False, key="pipeline_inject_noise")
    noise_class = None
    target_snr = 10.0

    if inject_noise:
        noise_class = st.sidebar.selectbox("Background Noise Event", CLASSES, index=8, key="pipeline_noise_class")
        target_snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", min_value=-5.0, max_value=25.0, value=10.0, step=1.0, key="pipeline_snr")
        st.sidebar.caption("Lower SNR = louder street noise (0 dB = equal acoustic power)")

    num_speakers = st.sidebar.number_input("Expected Speakers (0 = Auto-Estimate)", min_value=0, max_value=8, value=2, key="pipeline_num_spk")

    # Main Tab Display
    if selected_file and selected_file.exists():
        try:
            waveform, audio_dur = load_full_audio(selected_file)
        except Exception as e:
            st.error(f"Failed to decode audio file: {e}")
            waveform = np.array([], dtype=np.float32)
            audio_dur = 0.0

        if audio_dur > 300.0:
            st.error("Audio duration exceeds 5-minute maximum limit for online analysis.")
        elif len(waveform) > 0:
            applied_noise_name = "None (Clean Audio)"
            current_snr = None

            if inject_noise:
                classes_arg = [noise_class] if noise_class is not None else None
                mixer = UrbanNoiseMixer(classes=classes_arg)
                waveform, applied_noise_name, current_snr = mixer.mix(waveform, snr_db=target_snr)

            # 1. Audio Player & Visualizer
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🎧 Audio Playback")
                buf = io.BytesIO()
                sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
                st.audio(buf.getvalue(), format="audio/wav")
                st.caption(f"Source: `{selected_file.name}` | Duration: **{audio_dur:.2f}s** | Noise: **{applied_noise_name}**" + (f" @ {current_snr:.1f} dB" if current_snr is not None else ""))

            with col2:
                st.subheader("📊 Audio Waveform & Spectrogram")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.2), sharex=False)
                time_axis = np.linspace(0, len(waveform) / SAMPLE_RATE, len(waveform))
                ax1.plot(time_axis, waveform, color="#1E88E5", lw=0.8)
                ax1.set_ylabel("Amp")
                ax1.set_xlim(0, len(waveform) / SAMPLE_RATE)
                ax1.grid(True, alpha=0.3)

                spec = compute_log_mel_spectrogram(waveform)
                ax2.imshow(spec, aspect="auto", origin="lower", cmap="magma")
                ax2.set_xlabel("Time Frames")
                ax2.set_ylabel("Mel Freq")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")

            # 2. Multi-Task Unified Analysis
            if allow_demo:
                st.markdown(
                    '<div class="demo-banner">⚠️ <b>Demo Mode Active:</b> Predictions are produced using synthetic unverified model weights for pipeline testing.</div>',
                    unsafe_allow_html=True,
                )

            if (not analyzer_ready or analyzer is None) and not allow_demo:
                st.warning("Cannot run analysis: Required model checkpoints are missing. Train checkpoints or enable 'Synthetic Demo Mode' in the sidebar.")
            elif analyzer is not None:
                if st.button("🚀 Run Multi-Task Audio Intelligence Pipeline", type="primary", key="pipeline_run_btn"):
                    device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                    with st.spinner(f"Executing neural multi-task inference on {device_str}..."):
                        spk_arg = None if num_speakers == 0 else int(num_speakers)
                        file_label = selected_file.name if selected_file else "custom_waveform"
                        report = analyzer.analyze_waveform(waveform, file_name=file_label, num_speakers=spk_arg)

                        # Metric Cards
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Duration", report["duration"])
                        with m2:
                            st.metric("Detected Noise Scene", report["environmental_acoustic_scene"]["detected_class"].replace("_", " ").title())
                        with m3:
                            st.metric("Scene Confidence", f"{report['environmental_acoustic_scene']['confidence']*100:.1f}%")
                        with m4:
                            st.metric("Speech Turns", report["total_speech_turns"])

                        st.markdown("### 🗣️ Continuous Speaker Diarization & Emotion Timeline")
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

                        st.caption("ℹ️ *Disclaimer: Emotion recognition reflects acoustic vocal characteristics and does not provide definitive proof of internal emotional states.*")

                        with st.expander("📄 View Full JSON Ruflo Provenance Report"):
                            st.json(report)

        # Cleanup uploaded temporary audio if one was created
        if is_temp_upload and selected_file and selected_file.exists():
            try:
                os.unlink(selected_file)
            except Exception:
                pass
    else:
        st.info("Please select or upload an audio file from the left sidebar to begin.")


# ==============================================================================
# TAB 2: SPEAKER STUDIO & VOICEPRINT VERIFICATION
# ==============================================================================
with tab_speaker_studio:
    st.header("🆔 Speaker Identification & Voiceprint Verification Studio")
    st.markdown(
        "Leverages the **VoxCeleb Speaker Embedding Network** (`SpeakerEmbeddingNet` with Temporal Stats Pooling), "
        "producing compact 128-dimensional L2-normalized voiceprint representations."
    )

    studio_col1, studio_col2 = st.columns([1, 1])

    # Section A: Enrolled Voiceprint Registry
    with studio_col1:
        st.subheader("📋 Enrolled Voiceprint Registry")
        enrolled = st.session_state.get("session_enrolled_speakers", {})
        if enrolled:
            for name, emb in enrolled.items():
                with st.expander(f"👤 **{name}** (128-d Vector)", expanded=False):
                    st.caption(f"Norm: {np.linalg.norm(emb):.4f} | First 8 dims: `{np.round(emb[:8], 3)}`")
                    fig_emb = plt.figure(figsize=(6, 1.2))
                    ax_emb = fig_emb.add_subplot(111)
                    ax_emb.bar(range(128), emb, color="#1E88E5", width=0.8)
                    ax_emb.set_xlim(-1, 128)
                    ax_emb.set_ylabel("Value")
                    ax_emb.set_xlabel("Embedding Dimension (0-127)")
                    ax_emb.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_emb)
                    plt.close(fig_emb)
        else:
            st.info("No speakers currently enrolled in session.")

        st.markdown("#### ➕ Enroll New Speaker")
        new_spk_name = st.text_input("Speaker Label (Alphanumeric)", value="Speaker_Custom", key="new_spk_name")
        enroll_upload = st.file_uploader("Upload Reference Audio (.wav, Max 25MB)", type=["wav"], key="enroll_upload")

        if st.button("Enroll Voiceprint into Registry", key="enroll_btn"):
            sanitized_name = re.sub(r"[^a-zA-Z0-9_\-]", "", new_spk_name.strip())
            if not sanitized_name:
                st.error("Speaker label must contain valid alphanumeric characters (a-z, A-Z, 0-9, _, -).")
            elif enroll_upload is None:
                st.error("Please upload an audio file to extract reference voiceprint.")
            elif not analyzer_ready or analyzer is None:
                st.error("Speaker embedding model not loaded. Enable Synthetic Demo Mode or train checkpoint.")
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_enroll:
                    tmp_enroll.write(enroll_upload.getbuffer())
                    tmp_path = Path(tmp_enroll.name)
                try:
                    ref_wave, _ = load_full_audio(tmp_path)
                    emb = analyzer.speaker_identifier.extract_embedding(ref_wave)
                    st.session_state["session_enrolled_speakers"][sanitized_name] = emb
                    st.success(f"✅ Successfully enrolled voiceprint for **{sanitized_name}** in session!")
                    if hasattr(st, "rerun"):
                        getattr(st, "rerun")()
                    elif hasattr(st, "experimental_rerun"):
                        getattr(st, "experimental_rerun")()
                except Exception as e:
                    st.error(f"Error during enrollment: {e}")
                finally:
                    if tmp_path.exists():
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

    # Section B: Pairwise 1-to-1 Voiceprint Verification
    with studio_col2:
        st.subheader("🔍 1-to-1 Pairwise Voiceprint Verification")
        st.markdown("Compare two audio recordings to determine if they originate from the **same speaker**.")

        vox_root = BASE_DIR / "wav"
        vox_files = list(vox_root.glob("*/*/*.wav"))[:40] if vox_root.exists() else []
        vox_labels = [f"{f.parent.parent.name}/{f.name}" for f in vox_files]

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("**Voice Sample A**")
            idx_a = st.selectbox("Sample A", range(len(vox_labels)), format_func=lambda i: vox_labels[i], index=0 if len(vox_labels) > 0 else 0, key="v_sample_a")
        with c_v2:
            st.markdown("**Voice Sample B**")
            idx_b = st.selectbox("Sample B", range(len(vox_labels)), format_func=lambda i: vox_labels[i], index=min(1, len(vox_labels)-1) if len(vox_labels) > 1 else 0, key="v_sample_b")

        threshold = st.slider("Cosine Verification Threshold", min_value=0.20, max_value=0.90, value=0.60, step=0.05, key="verify_thresh")

        if st.button("🔬 Compute Pairwise Verification", type="primary", key="verify_btn") and vox_files and idx_a is not None and idx_b is not None:
            if not analyzer_ready or analyzer is None:
                st.error("Model checkpoints unavailable. Enable Synthetic Demo Mode or train checkpoint.")
            else:
                file_a = vox_files[idx_a]
                file_b = vox_files[idx_b]

                wav_a, _ = load_full_audio(file_a)
                wav_b, _ = load_full_audio(file_b)

                emb_a = analyzer.speaker_identifier.extract_embedding(wav_a)
                emb_b = analyzer.speaker_identifier.extract_embedding(wav_b)

                # Cosine similarity between L2-normalized embeddings
                cosine_sim = float(np.dot(emb_a, emb_b))
                is_match = cosine_sim >= threshold

                st.markdown("---")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                with m_col2:
                    st.metric("Threshold", f"{threshold:.2f}")

                # Verification Verdict
                if is_match:
                    st.markdown(
                        f'<div class="verdict-match">'
                        f'<h4>✅ VERIFICATION CONFIRMED: MATCH</h4>'
                        f'<p>Cosine similarity ({cosine_sim:.4f}) meets or exceeds calibrated threshold ({threshold:.2f}). Both utterances match acoustic voiceprints.</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="verdict-diff">'
                        f'<h4>❌ VERIFICATION REJECTED: IMPOSTOR / DIFFERENT SPEAKER</h4>'
                        f'<p>Cosine similarity ({cosine_sim:.4f}) falls below threshold ({threshold:.2f}). Acoustic voiceprints indicate distinct speakers.</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Visual spectrogram comparison
                st.markdown("#### Side-by-Side Spectrogram Comparison")
                fig_cmp, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 2.5))
                spec_a = compute_log_mel_spectrogram(wav_a)
                spec_b = compute_log_mel_spectrogram(wav_b)
                ax_a.imshow(spec_a, aspect="auto", origin="lower", cmap="magma")
                ax_a.set_title(f"Sample A: {file_a.parent.parent.name}")
                ax_a.set_xlabel("Time Frames")
                ax_b.imshow(spec_b, aspect="auto", origin="lower", cmap="magma")
                ax_b.set_title(f"Sample B: {file_b.parent.parent.name}")
                ax_b.set_xlabel("Time Frames")
                plt.tight_layout()
                st.pyplot(fig_cmp)
                plt.close(fig_cmp)


# ==============================================================================
# TAB 3: MODEL BENCHMARKS & EXPERIMENT TRACKING
# ==============================================================================
with tab_metrics:
    st.header("📈 Model Benchmarks, Cross-Validation & Provenance")
    st.markdown(
        "Empirical performance metrics recorded across cross-validation folds and held-out evaluation sets."
    )

    experiments_path = BASE_DIR / "pipeline_output" / "ruflo_experiment_history.json"
    if not experiments_path.exists():
        experiments_path = BASE_DIR / "pipeline_output" / "ruflo_experiments.json"

    if not experiments_path.exists():
        st.warning("No experiment records found in `pipeline_output/ruflo_experiment_history.json`. Run training pipelines to populate benchmark results.")
    else:
        try:
            with open(experiments_path, "r", encoding="utf-8") as f:
                exp_data = json.load(f)

            runs = exp_data.get("runs", [])
            urban_runs = [r for r in runs if "urban" in r.get("experiment", "").lower() or "urban" in r.get("run_name", "").lower() or "fold" in str(r.get("test_fold", ""))]
            vox_runs = [r for r in runs if "vox" in r.get("experiment", "").lower() or "speaker" in r.get("experiment", "").lower() or "vox" in r.get("run_name", "").lower()]
            emotion_runs = [r for r in runs if "emotion" in r.get("experiment", "").lower() or "emotion" in r.get("run_name", "").lower()]

            col_u1, col_u2 = st.columns([1, 1])

            with col_u1:
                st.subheader("🏙️ UrbanSound8K 10-Fold Cross-Validation")
                if urban_runs:
                    accs = [r["val_accuracy"] * 100 for r in urban_runs if "val_accuracy" in r]
                    f1s = [r["val_macro_f1"] for r in urban_runs if "val_macro_f1" in r]
                    mean_acc = np.mean(accs) if accs else 0.0
                    mean_f1 = np.mean(f1s) if f1s else 0.0
                    std_f1 = np.std(f1s) if len(f1s) > 1 else 0.0

                    m_u1, m_u2, m_u3 = st.columns(3)
                    with m_u1:
                        st.metric("Total Folds", len(urban_runs))
                    with m_u2:
                        st.metric("Mean Val Accuracy", f"{mean_acc:.2f}%")
                    with m_u3:
                        st.metric("Mean Macro-F1", f"{mean_f1:.4f} ± {std_f1:.4f}")

                    # Table of Folds
                    table_rows = []
                    for r in sorted(urban_runs, key=lambda x: x.get("test_fold", 0)):
                        val_acc = float(r.get("val_accuracy") or 0.0)
                        val_f1 = float(r.get("val_macro_f1") or 0.0)
                        hyp = r.get("hyperparameters", {})
                        model_label = hyp.get("model_type", "AudioCNN")
                        if hyp.get("use_cmvn", False):
                            model_label += " + CMVN"
                        table_rows.append({
                            "Fold": f"Fold {r.get('test_fold')}",
                            "Model": model_label,
                            "Epochs": r.get("epochs"),
                            "Val Accuracy": f"{val_acc*100:.2f}%",
                            "Macro-F1": f"{val_f1:.4f}",
                            "Checkpoint": Path(str(r.get("checkpoint", ""))).name,
                        })
                    st.dataframe(table_rows, use_container_width=True)

                    # Bar Chart of Cross-Validation Folds
                    fig_cv = plt.figure(figsize=(6, 2.8))
                    ax_cv = fig_cv.add_subplot(111)
                    folds_idx = [r.get("test_fold", i+1) for i, r in enumerate(sorted(urban_runs, key=lambda x: x.get("test_fold", 0)))]
                    f1_vals = [r.get("val_macro_f1", 0) for r in sorted(urban_runs, key=lambda x: x.get("test_fold", 0))]
                    ax_cv.bar(folds_idx, f1_vals, color="#1E88E5", alpha=0.85, width=0.6)
                    ax_cv.axhline(mean_f1, color="#D32F2F", linestyle="--", label=f"Mean F1 ({mean_f1:.3f})")
                    ax_cv.set_xticks(folds_idx)
                    ax_cv.set_xlabel("Cross-Validation Fold")
                    ax_cv.set_ylabel("Macro-F1 Score")
                    ax_cv.set_ylim(0, 1.0)
                    ax_cv.legend(loc="lower right")
                    ax_cv.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_cv)
                    plt.close(fig_cv)

            with col_u2:
                st.subheader("🗣️ VoxCeleb Speaker Network")
                if vox_runs:
                    v_run = vox_runs[-1]
                    m_v1, m_v2, m_v3 = st.columns(3)
                    with m_v1:
                        vox_top1 = float(v_run.get('val_accuracy') or 0.0)
                        st.metric("Closed-Set Top-1", f"{vox_top1*100:.2f}%")
                    with m_v2:
                        per_c = v_run.get('per_class_f1', {})
                        top5 = per_c.get('top5_accuracy', v_run.get('top5_accuracy'))
                        if top5 is not None:
                            st.metric("Top-5 Accuracy", f"{float(top5)*100:.2f}%")
                        else:
                            st.metric("Top-5 Accuracy", "Not measured")
                    with m_v3:
                        st.metric("Embedding Dim", "128 (L2)")

                    st.caption("ℹ️ *Note: Evaluated with Additive Angular Margin (ArcFace) loss and temporal stats pooling.*")
                    st.markdown("**Recorded Training Hyperparameters:**")
                    params = v_run.get("hyperparameters", {})
                    st.json(params)

                st.subheader("🎭 RAVDESS Speech Emotion Recognition")
                if emotion_runs:
                    e_run = emotion_runs[-1]
                    e_f1 = float(e_run.get("val_macro_f1", 0.0))
                    e_acc = float(e_run.get("val_accuracy", 0.0))
                    m_e1, m_e2 = st.columns(2)
                    with m_e1:
                        st.metric("SER Val Macro-F1", f"{e_f1:.4f}")
                    with m_e2:
                        st.metric("SER Val Accuracy", f"{e_acc*100:.2f}%")
                else:
                    st.markdown(
                        "Evaluated across 8 emotional classes on speaker-independent actor holdout:\n"
                        "- Classes: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`\n"
                        "- Validation Split: Actors 19-20 (checkpoint selection)\n"
                        "- Test Split: Actors 21-24 (independent holdout)\n"
                        "- Active Checkpoint: `best_emotion_model.pt`"
                    )

            # A/B Experiment Comparison Panel
            st.divider()
            st.subheader("🔬 A/B Experiment Lineage & Architecture Benchmarks")
            st.markdown(
                "Compare baseline architectures against upgraded convolutional backbones, margin losses, and normalization schemes."
            )
            ab_comparisons = [
                {
                    "Component": "Speaker Identification & Diarization",
                    "Baseline Architecture": "Baseline Conv2D + Cosine CE",
                    "Upgraded Architecture": "SE-ResNet Backbone + ArcFace Margin (m=0.30, s=30.0)",
                    "Key Metric": "Cluster tightness & Inter-class Angular Separation",
                    "Inference Overhead": "Zero parameter overhead at runtime (128-d vector)",
                },
                {
                    "Component": "Environmental Audio Classifier",
                    "Baseline Architecture": "AudioCNN (Global Log-Mel Normalization)",
                    "Upgraded Architecture": "SEAudioCNN + Per-Channel CMVN Normalization",
                    "Key Metric": "Noise-robust Macro-F1 across background stationary noise",
                    "Inference Overhead": "+4 SE Squeeze-and-Excitation Attention Blocks (~2% latency)",
                },
                {
                    "Component": "Speech Emotion Recognition (SER)",
                    "Baseline Architecture": "EmotionClassifierCNN (Handcrafted Mel-Spectrogram)",
                    "Upgraded Architecture": "SSLEmotionClassifier (Pretrained wav2vec 2.0 / HuBERT Adapter)",
                    "Key Metric": "Speech Invariance & Cross-Actor Generalization",
                    "Inference Overhead": "Frozen upstream transformer with lightweight attentive pooling",
                },
            ]
            st.dataframe(ab_comparisons, use_container_width=True)

        except Exception as e:
            st.error(f"Error parsing experiment history: {e}")

