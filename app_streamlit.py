#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:13:57 2025

@author: jason
"""

"""
Streamlit UI for Satellite Change Detection

Run:
  pip install streamlit torch torchvision opencv-python pillow numpy tqdm scikit-image
  streamlit run app_streamlit.py

This app expects a checkpoint produced by satellite_change_detection.py (model.pt) and two
co-registered images (same size). It will render:
- Binary mask
- Overlay on before/after
- Probability heatmap
- RGB absdiff / gray absdiff / masked absdiff
"""

import io
import numpy as np
import torch
import streamlit as st
from PIL import Image
import cv2

# Import utilities from your training/prediction script in the same CWD
from satellite_change_detection import (
    SiameseUNet,
    predict_prob,
    normalize,
    align_image,
    morphological_cleanup,
    absdiff_rgb,
    absdiff_gray,
    make_overlay,
    apply_colormap,
)

st.set_page_config(page_title="Satellite Change Detection", layout="wide")

st.title("🛰️ Satellite Change Detection — Streamlit UI")
st.write("Upload a trained checkpoint (model.pt) and a before/after pair. Adjust threshold and options, then click **Predict**.")

with st.sidebar:
    st.header("Model & Inference Settings")
    ckpt_file = st.file_uploader("Checkpoint (model.pt)", type=["pt", "pth"], accept_multiple_files=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Compute device: **{device.upper()}**")
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    min_blob = st.number_input("Min blob (px)", min_value=0, max_value=100000, value=50, step=10)
    tile = st.number_input("Tile size (0 = whole image)", min_value=0, max_value=4096, value=512, step=32)
    overlap = st.number_input("Overlap (px)", min_value=0, max_value=512, value=32, step=8)
    align = st.checkbox("ECC align AFTER to BEFORE", value=False)

    st.header("Visualization")
    overlay_on = st.selectbox("Overlay on", options=["before", "after"], index=0)
    overlay_alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05)
    outline = st.checkbox("Draw outline", value=True)

colA, colB = st.columns(2)
with colA:
    before_file = st.file_uploader("Before image", type=["png","jpg","jpeg","tif","tiff"]) 
with colB:
    after_file  = st.file_uploader("After image",  type=["png","jpg","jpeg","tif","tiff"]) 

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(ckpt_bytes: bytes, device: str):
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    in_ch = int(ckpt.get("in_ch", 3))
    base = int(ckpt.get("base", 32))
    model = SiameseUNet(in_ch=in_ch, base=base)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, in_ch


def _np_from_upload(file) -> np.ndarray:
    img = Image.open(file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _encode_png_rgb(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        ok, buf = cv2.imencode(".png", arr)
        return buf.tobytes()
    # assume RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()

predict_clicked = st.button("🚀 Predict", type="primary")

if predict_clicked:
    if not ckpt_file:
        st.error("Please upload a model checkpoint (model.pt).")
    elif not before_file or not after_file:
        st.error("Please upload both before and after images.")
    else:
        with st.spinner("Loading model and preparing images..."):
            model, in_ch = load_model_from_bytes(ckpt_file.getvalue(), device)
            a = _np_from_upload(before_file)
            b = _np_from_upload(after_file)
            # Ensure same size
            if a.shape != b.shape:
                st.error(f"Image size mismatch: before {a.shape} vs after {b.shape}. Please upload co-registered images of the same size.")
                st.stop()
            a = normalize(a)
            b = normalize(b)
            if align:
                b = align_image(a, b)
        with st.spinner("Running inference..."):
            prob = predict_prob(model, a, b, tile=int(tile), overlap=int(overlap), device=device)
            mask = (prob >= float(thr)).astype(np.uint8) * 255
            mask = morphological_cleanup(mask, min_blob=int(min_blob))
            overlay_base = a if overlay_on == "before" else b
            overlay = make_overlay(overlay_base, mask, color=(255,0,0), alpha=float(overlay_alpha), outline=bool(outline), thickness=2)
            heat = apply_colormap(prob)
            d_rgb = absdiff_rgb(a,b)
            d_gray = absdiff_gray(a,b)
            d_masked = (d_rgb * ((mask>0).astype(np.uint8)[...,None])).astype(np.uint8)

        st.success("Done!")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(a, caption="Before", use_column_width=True)
        with c2:
            st.image(b, caption="After", use_column_width=True)
        with c3:
            st.image(mask, caption=f"Binary mask (thr={thr:.2f}, min_blob={int(min_blob)})", use_column_width=True, clamp=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            st.image(overlay, caption=f"Overlay on {overlay_on}", use_column_width=True)
        with c5:
            st.image(heat, caption="Probability heatmap", use_column_width=True)
        with c6:
            st.image(d_rgb, caption="RGB absdiff", use_column_width=True)

        c7, c8 = st.columns(2)
        with c7:
            st.image(d_gray, caption="Gray absdiff", use_column_width=True, clamp=True)
        with c8:
            st.image(d_masked, caption="Masked absdiff (within predicted change)", use_column_width=True)

        st.divider()
        st.caption("Downloads")
        colD1, colD2, colD3, colD4, colD5 = st.columns(5)
        with colD1:
            st.download_button("mask.png", _encode_png_rgb(mask), file_name="mask.png", mime="image/png")
        with colD2:
            st.download_button("overlay.png", _encode_png_rgb(overlay), file_name="overlay.png", mime="image/png")
        with colD3:
            st.download_button("heatmap.png", _encode_png_rgb(heat), file_name="heatmap.png", mime="image/png")
        with colD4:
            st.download_button("diff_rgb.png", _encode_png_rgb(d_rgb), file_name="diff_rgb.png", mime="image/png")
        with colD5:
            st.download_button("diff_gray.png", _encode_png_rgb(d_gray), file_name="diff_gray.png", mime="image/png")

        # Also offer masked diff
        st.download_button("diff_masked.png", _encode_png_rgb(d_masked), file_name="diff_masked.png", mime="image/png")
