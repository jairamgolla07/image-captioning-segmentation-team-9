import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st  # type: ignore
import numpy as np
from PIL import Image

from src.captioning.captioning import ImageCaptioner
from src.segmentation.segmentation import ImageSegmenter

st.set_page_config(
    page_title="Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Image Captioning + Segmentation")

@st.cache_resource(show_spinner=False)
def load_models():
    captioner = ImageCaptioner()
    segmenter = ImageSegmenter()
    return captioner, segmenter

captioner, segmenter = load_models()

uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)          # ← updated

    with st.spinner("Generating caption and masks..."):
        caption = captioner.caption(image)
        masks, labels, scores = segmenter.segment(image)

    st.subheader("📌 Caption")
    st.success(caption)

    # Build mask overlay
    alpha = 0.4
    rng = np.random.default_rng(42)
    colors = rng.random((len(masks), 3))
    img_np = np.asarray(image).astype(float) / 255.0

    for mask_tensor, color in zip(masks, colors):
        mask = mask_tensor.squeeze(0).numpy()
        colored = np.zeros_like(img_np)
        colored[mask > 0.5] = color
        img_np = np.where(mask[..., None], img_np * (1 - alpha) + colored * alpha, img_np)

    st.subheader("🎯 Segmentation Result")
    st.image((img_np * 255).astype(np.uint8), use_container_width=True)         # ← updated

    st.subheader("📋 Detected Objects")
    for lbl, sc in zip(labels, scores):
        st.write(f"Class ID: {int(lbl)} • Score: {sc:.2f}")
