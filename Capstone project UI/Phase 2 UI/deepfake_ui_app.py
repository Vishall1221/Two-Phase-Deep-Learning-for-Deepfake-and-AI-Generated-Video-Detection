from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ensemble_video_predictor import DeepfakeEnsemblePredictor


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHITECTURE_IMAGE = PROJECT_ROOT / "assets" / "final_best_ensemble_architecture.png"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"


st.set_page_config(
    page_title="Deepfake Ensemble Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(0, 163, 255, 0.18), transparent 32%),
                    radial-gradient(circle at top right, rgba(255, 94, 125, 0.16), transparent 30%),
                    linear-gradient(180deg, #07111f 0%, #0d1628 52%, #09111b 100%);
                color: #edf3ff;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero-card, .panel-card, .metric-card, .result-card {
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(9, 18, 35, 0.78);
                backdrop-filter: blur(12px);
                border-radius: 22px;
                box-shadow: 0 22px 60px rgba(0,0,0,0.18);
            }
            .hero-card {
                padding: 1.4rem 1.6rem;
                margin-bottom: 1rem;
            }
            .panel-card {
                padding: 1rem 1.2rem;
            }
            .metric-card {
                padding: 1rem 1.15rem;
                min-height: 118px;
            }
            .result-card {
                padding: 1.25rem 1.4rem;
                margin-bottom: 1rem;
            }
            .eyebrow {
                color: #90b7ff;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                font-size: 0.78rem;
                font-weight: 700;
            }
            .hero-title {
                font-size: 2.35rem;
                font-weight: 800;
                line-height: 1.05;
                margin: 0.2rem 0 0.6rem 0;
            }
            .hero-subtitle {
                color: #d4deef;
                font-size: 1.02rem;
                line-height: 1.65;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.85rem;
            }
            .badge {
                border-radius: 999px;
                padding: 0.38rem 0.78rem;
                border: 1px solid rgba(255,255,255,0.1);
                background: rgba(255,255,255,0.05);
                font-size: 0.88rem;
                color: #edf3ff;
            }
            .metric-title {
                color: #9cb3d9;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.55rem;
            }
            .metric-value {
                font-size: 1.95rem;
                font-weight: 800;
                line-height: 1;
                margin-bottom: 0.45rem;
            }
            .metric-note {
                color: #d4deef;
                font-size: 0.92rem;
                line-height: 1.55;
            }
            .result-real {
                border-color: rgba(37, 211, 102, 0.35);
                box-shadow: 0 22px 60px rgba(17, 63, 39, 0.18);
            }
            .result-fake {
                border-color: rgba(255, 94, 125, 0.38);
                box-shadow: 0 22px 60px rgba(89, 19, 35, 0.2);
            }
            .result-label {
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .result-score {
                color: #d6e0f1;
                font-size: 1rem;
            }
            .simple-note {
                color: #d4deef;
                font-size: 0.96rem;
                line-height: 1.7;
            }
            div[data-testid="stFileUploader"] {
                background: rgba(255,255,255,0.03);
                border: 1px dashed rgba(160, 189, 255, 0.28);
                border-radius: 18px;
                padding: 0.4rem 0.5rem;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: transparent;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_predictor() -> DeepfakeEnsemblePredictor:
    return DeepfakeEnsemblePredictor(runtime_root=RUNTIME_ROOT, require_cuda=True)


def save_upload(uploaded_file) -> Path:
    upload_root = RUNTIME_ROOT / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    destination = upload_root / f"{timestamp}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result: dict[str, object]) -> None:
    ensemble = result["ensemble"]
    predicted_label = ensemble["predicted_label"]
    fake_score = float(ensemble["video_probability"])
    confidence = float(ensemble["confidence"])
    card_class = "result-card result-fake" if predicted_label == "Deepfake" else "result-card result-real"
    accent = "#ff7b93" if predicted_label == "Deepfake" else "#47d17f"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="eyebrow">Final Ensemble Decision</div>
            <div class="result-label" style="color: {accent};">{predicted_label}</div>
            <div class="result-score">Fake probability: <strong>{fake_score:.2%}</strong> &nbsp;|&nbsp; Confidence: <strong>{confidence:.2%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_details(result: dict[str, object]) -> None:
    extraction = result["face_extraction"]
    models = result["model_predictions"]
    ensemble = result["ensemble"]

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Final Test Benchmark", f"{float(result['ensemble_test_accuracy']):.2%}", "Frozen best ensemble accuracy")
    with metric_columns[1]:
        render_metric_card("Frames Used", str(len(extraction["face_paths"])), "Fifteen face crops per uploaded video")
    with metric_columns[2]:
        render_metric_card(
            "Faces Detected",
            str(extraction["detected_faces_before_padding"]),
            f"Padding duplicates: {extraction['padding_applied']}",
        )
    with metric_columns[3]:
        render_metric_card("Device", result["device_name"], "Inference runs on CUDA in this UI")

    st.markdown("### Model Branch Scores")
    branch_columns = st.columns(len(models))
    for column, model in zip(branch_columns, models):
        with column:
            render_metric_card(
                model["model_name"],
                f"{float(model['video_probability']):.2%}",
                f"Weight {float(model['weight']):.2f} | Input {model['image_size']}px",
            )

    frame_df = pd.DataFrame(
        {
            "Frame": list(range(1, len(ensemble["frame_probabilities"]) + 1)),
            "Ensemble": ensemble["frame_probabilities"],
            **{model["model_name"]: model["frame_probabilities"] for model in models},
        }
    )
    st.markdown("### Frame-Level Probability Trend")
    st.line_chart(frame_df.set_index("Frame"), height=320)

    st.markdown("### Extracted Face Previews")
    st.image(
        extraction["face_paths"][:8],
        width=150,
        caption=[f"Face {idx + 1}" for idx in range(min(8, len(extraction["face_paths"])))],
    )

    with st.expander("Inference Report JSON", expanded=False):
        st.json(result)
        st.download_button(
            "Download prediction.json",
            data=json.dumps(result, indent=2),
            file_name="prediction.json",
            mime="application/json",
        )


def main() -> None:
    inject_styles()
    predictor = load_predictor()

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Generalized Deepfake Reduction UI</div>
            <div class="hero-title">Upload a video and predict <br/>Real vs Deepfake with the frozen 96.67% ensemble.</div>
            <div class="hero-subtitle">
                This interface reuses the same best-performing late-fusion pipeline from your experiments:
                face extraction with MTCNN, branch inference with <strong>Xception</strong> and <strong>EfficientNet-B2</strong>,
                then weighted ensemble fusion for the final verdict.
            </div>
            <div class="badge-row">
                <div class="badge">Xception weight: 0.45</div>
                <div class="badge">EfficientNet-B2 weight: 0.55</div>
                <div class="badge">GPU: {predictor.device_name}</div>
                <div class="badge">15 face crops per video</div>
                <div class="badge">Best test accuracy: {predictor.ensemble_test_accuracy:.2%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Final Deployment Model")
        st.write("Frozen best ensemble from your experiments.")
        st.write(f"- Validation accuracy: `{predictor.ensemble_validation_accuracy:.2%}`")
        st.write(f"- Test accuracy: `{predictor.ensemble_test_accuracy:.2%}`")
        st.write(f"- Device: `{predictor.device_name}`")
        if ARCHITECTURE_IMAGE.exists():
            st.image(str(ARCHITECTURE_IMAGE), caption="Final late-fusion architecture")
        st.markdown("### Processing Flow")
        st.write("1. Upload video")
        st.write("2. Sample frames and extract faces with MTCNN")
        st.write("3. Run Xception and EfficientNet-B2 branches")
        st.write("4. Fuse scores and return Real / Deepfake")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a video for inference",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="The app will extract 15 face crops and run the best ensemble on GPU.",
        )
        if uploaded_file is not None:
            st.video(uploaded_file)
        analyze = st.button("Run Deepfake Analysis", type="primary", use_container_width=True, disabled=uploaded_file is None)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### What this UI uses")
        st.write("- Same merged-data deployment path as your best experiment")
        st.write("- MTCNN face extraction settings matched to dataset preprocessing")
        st.write("- Xception checkpoint from the `95.56%` single-model run")
        st.write("- EfficientNet-B2 checkpoint from the frozen best ensemble")
        st.write("- Final late fusion at `0.45 / 0.55`")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze and uploaded_file is not None:
        saved_video_path = save_upload(uploaded_file)
        progress_bar = st.progress(0)
        status_box = st.empty()

        def update_status(message: str, progress: float | None) -> None:
            status_box.info(message)
            if progress is not None:
                progress_bar.progress(max(0.0, min(progress, 1.0)))

        try:
            result = predictor.predict_video(saved_video_path, progress_callback=update_status)
        except Exception as exc:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Inference failed: {exc}")
            return

        progress_bar.progress(1.0)
        status_box.success("Inference complete.")
        render_result_card(result)
        render_prediction_details(result)


if __name__ == "__main__":
    main()
