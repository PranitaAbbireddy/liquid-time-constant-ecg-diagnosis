import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.captum_wrappers import ECGCaptumWrapper
from src.models.factory import build_model
from src.preprocessing.normalize import TrainStatisticsNormalizer
from src.utils.config import load_config
from src.utils.paths import resolve_path

st.set_page_config(page_title="LNN ECG Diagnosis", layout="wide")

st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #5f6368; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
class_names = {int(k): v for k, v in config["classes"].items()}
paths = config["paths"]


@st.cache_resource
def load_checkpoint():
    checkpoint_path = resolve_path(f"{paths['checkpoints_dir']}/{config['training']['checkpoint_name']}")
    if not checkpoint_path.exists():
        return None, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", config)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = None
    if checkpoint.get("normalizer"):
        normalizer = TrainStatisticsNormalizer.from_dict(checkpoint["normalizer"])
    bias = checkpoint.get("logit_bias")
    logit_bias = np.array(bias, dtype=np.float32) if bias is not None else None
    return model, normalizer, device, logit_bias


def load_metrics():
    metrics_path = resolve_path(f"{paths['reports_dir']}/metrics/test_metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


_loaded = load_checkpoint()
if _loaded[0] is None:
    model, normalizer, device, logit_bias = None, None, torch.device("cpu"), None
else:
    model, normalizer, device, logit_bias = _loaded

st.markdown('<p class="main-title">Liquid Neural Network ECG Diagnosis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Research demo for automated arrhythmia beat classification on MIT-BIH style data.</p>',
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset",
        "Model",
        "Evaluation",
        "ECG Prediction",
        "Explainability",
        "Technical Details",
    ],
)

if section == "Overview":
    st.header("Overview")
    st.write(
        """
        An electrocardiogram (ECG) records the electrical activity of the heart over time.
        This project classifies single-beat ECG snippets into arrhythmia-related categories.

        The model uses a **liquid time-constant (LTC)** recurrent architecture designed for
        sequential signals. This interface is for education and research demonstration only
        and is **not** a medical device.
        """
    )

elif section == "Dataset":
    st.header("Dataset")
    st.write(
        """
        Training and evaluation use preprocessed MIT-BIH beat records (187 samples per beat, 5 classes).
        Class frequencies are highly imbalanced; macro-F1 is more informative than accuracy alone.
        """
    )
    img = resolve_path(f"{paths['graphs_dir']}/dataset/class_distribution.png")
    if img.exists():
        st.image(str(img), caption="Training class distribution")
    sample_img = resolve_path(f"{paths['graphs_dir']}/dataset/sample_beats.png")
    if sample_img.exists():
        st.image(str(sample_img), caption="Representative beats per class")

elif section == "Model":
    st.header("Model")
    with st.expander("Simple explanation", expanded=True):
        st.write(
            """
            The network reads the ECG beat sample-by-sample and updates an internal state using
            liquid dynamics with learnable time constants. The final state is mapped to class probabilities.
            """
        )
    with st.expander("Technical details"):
        st.json(config["model"])
        if model is not None:
            st.write(f"Trainable parameters: **{model.count_parameters():,}**")

elif section == "Evaluation":
    st.header("Evaluation")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No evaluation metrics found. Run `python scripts/train.py` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        col3.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
        col4.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")

        table_path = resolve_path(f"{paths['reports_dir']}/tables/per_class_metrics.csv")
        if table_path.exists():
            st.subheader("Per-class metrics")
            st.dataframe(pd.read_csv(table_path), use_container_width=True)

        cm_img = resolve_path(f"{paths['graphs_dir']}/evaluation/confusion_matrix.png")
        roc_img = resolve_path(f"{paths['graphs_dir']}/evaluation/roc_curves.png")
        loss_img = resolve_path(f"{paths['graphs_dir']}/training/loss_curves.png")
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion matrix (test)")
        if roc_img.exists():
            st.image(str(roc_img), caption="ROC curves (OvR)")
        if loss_img.exists():
            st.image(str(loss_img), caption="Training curves")

elif section == "ECG Prediction":
    st.header("ECG Prediction")
    if model is None:
        st.error("Checkpoint not found. Train the model with `python scripts/train.py`.")
    else:
        st.write("Upload a CSV row with 187 features, or 188 columns (index + features + optional label).")
        uploaded = st.file_uploader("ECG CSV", type=["csv"])
        example_path = resolve_path("Previous code/Interface/abnormal_patient_ecg.csv")
        use_example = st.button("Use bundled example beat")
        signal = None
        if uploaded is not None:
            df = pd.read_csv(uploaded, header=None)
            if df.shape[1] == 188:
                row = df.iloc[0, :-1].values.astype("float32")
            elif df.shape[1] == 187:
                row = df.iloc[0].values.astype("float32")
            else:
                st.error(f"Expected 187 or 188 columns, got {df.shape[1]}.")
                row = None
            if row is not None:
                signal = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
        elif use_example and example_path.exists():
            df = pd.read_csv(example_path, header=None)
            row = df.iloc[0, :-1].values.astype("float32")
            signal = torch.tensor(row, dtype=torch.float32).unsqueeze(0)

        if signal is not None:
            with torch.no_grad():
                x = signal.to(device)
                if normalizer is not None:
                    x = normalizer.transform_tensor(x)
                logits = model(x)
                if logit_bias is not None:
                    logits = logits + torch.tensor(logit_bias, device=device).unsqueeze(0)
                probs = softmax_probs(logits).squeeze().cpu().numpy()
            pred = int(probs.argmax())
            st.success(f"Predicted class: **{class_names[pred]}** ({pred})")
            st.bar_chart(
                pd.DataFrame({"probability": probs}, index=[class_names[i] for i in range(len(class_names))])
            )
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(signal.squeeze().cpu().numpy(), color="#4C72B0")
            ax.set_title("Input ECG beat")
            ax.set_xlabel("Time step")
            st.pyplot(fig)

elif section == "Explainability":
    st.header("Explainability")
    st.info(
        "Attribution maps highlight input regions associated with model output. "
        "They do not prove clinical causality or provide a diagnosis."
    )
    ig_img = resolve_path(f"{paths['graphs_dir']}/explainability/integrated_gradients_per_class.png")
    sal_img = resolve_path(f"{paths['graphs_dir']}/explainability/ig_vs_saliency.png")
    if ig_img.exists():
        st.image(str(ig_img), caption="Integrated Gradients per class")
    if sal_img.exists():
        st.image(str(sal_img), caption="Integrated Gradients vs Saliency")
    if not ig_img.exists():
        st.warning("Explainability figures not generated yet. Run training pipeline.")

elif section == "Technical Details":
    st.header("Technical Details")
    st.json(config)
    summary_path = resolve_path(f"{paths['reports_dir']}/summaries/run_summary.json")
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            st.json(json.load(f))
    st.write(f"PyTorch: `{torch.__version__}` | Device: `{device}`")
