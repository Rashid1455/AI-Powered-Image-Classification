"""Streamlit demo: upload a flower photo, get the class and a calibrated confidence.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gradcam import GradCAM, center_crop_view, overlay  # noqa: E402
from src.predict import Predictor  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Flower classifier", page_icon=":material/local_florist:", layout="wide")


@st.cache_resource(max_entries=4)
def load_predictor(model_path: str) -> Predictor:
    return Predictor(model_path)


@st.cache_resource(max_entries=4)
def load_gradcam(model_path: str) -> GradCAM:
    return GradCAM(load_predictor(model_path).model)


# ---------- sidebar: settings ----------
checkpoints = sorted(MODELS_DIR.glob("*.pt"))
if not checkpoints:
    st.error("No trained model found in `models/`. Run `python -m src.train` first.", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.header("Settings")
    names = [c.stem for c in checkpoints]
    default = names.index("mobilenet_v2") if "mobilenet_v2" in names else 0
    model_name = st.selectbox("Model", names, index=default)
    threshold = st.slider(
        "Uncertainty threshold", 0.0, 1.0, 0.60, 0.05,
        help="If the top confidence is below this value, the app answers *uncertain* "
             "instead of guessing.",
    )
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True,
                         help="Highlights the regions of the photo that drove the prediction.")

model_path = str(MODELS_DIR / f"{model_name}.pt")
predictor = load_predictor(model_path)
predictor.threshold = threshold

with st.sidebar:
    st.divider()
    st.caption("Classes this model knows")
    st.markdown(" · ".join(c.capitalize() for c in predictor.classes))
    st.caption(f"Architecture: `{predictor.arch}` · temperature: `{predictor.temperature:.2f}`")

# ---------- main ----------
st.title(":material/local_florist: Flower classifier")
st.caption(
    "Upload a photo of a flower. The model predicts which of the five species it is "
    "and how confident it is. Photos of anything else will still be forced into one of "
    "the five classes, which is why low-confidence answers are reported as *uncertain*."
)

files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                         accept_multiple_files=True)

if not files:
    st.info("Drop a JPG or PNG above to get started.", icon=":material/upload:")
    st.stop()

summary_rows = []
for f in files:
    try:
        img = Image.open(f).convert("RGB")
    except Exception:  # noqa: BLE001
        st.error(f"`{f.name}` could not be read as an image.", icon=":material/broken_image:")
        continue

    pred = predictor.predict(img)
    summary_rows.append({"file": f.name, "prediction": pred.label, "confidence": pred.confidence})

    with st.container(border=True):
        cols = st.columns([1, 1, 1.2] if show_cam else [1, 1.4])
        cols[0].image(img, caption=f.name, width="stretch", alt=f"Uploaded photo {f.name}")

        if show_cam:
            cam = load_gradcam(model_path)(predictor.preprocess(img))
            view = center_crop_view(img, predictor.cfg["image"]["resize_short"], predictor.cfg["image"]["size"])
            cols[1].image(overlay(view, cam), caption="Where the model looked", width="stretch",
                          alt=f"Grad-CAM heatmap for {f.name}")

        with cols[-1]:
            if pred.uncertain:
                st.subheader(":orange[:material/help: Uncertain]")
                st.markdown(f"Best guess **{pred.top_class.capitalize()}**, but only "
                            f"{pred.confidence:.1%} confident (threshold {threshold:.0%}).")
            else:
                st.subheader(f":green[:material/check_circle: {pred.top_class.capitalize()}]")
            st.metric("Confidence", f"{pred.confidence:.1%}")
            st.caption("Top predictions")
            for name, p in pred.top_k:
                st.progress(p, text=f"{name.capitalize()} · {p:.1%}")

if len(summary_rows) > 1:
    st.subheader("Batch summary")
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        column_config={"confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                                     format="percent")},
        alt="Table of predictions for every uploaded image",
    )
