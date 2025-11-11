import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import onnxruntime as ort
import os

# -------- Helper functions --------
def read_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img

def normalize(img):
    img = img.astype(np.float32) / 255.0
    return img

def to_tensor(img):
    img = normalize(img)
    return img.transpose(2,0,1)[None,...].astype(np.float32)

def predict_onnx(session, before, after, tile=512, overlap=32):
    H, W, _ = before.shape
    if tile <= 0 or (H <= tile and W <= tile):
        inputs = {"input_before": to_tensor(before), "input_after": to_tensor(after)}
        pred = session.run(None, inputs)[0][0,0]
        return pred
    # Sliding window
    out = np.zeros((H,W), dtype=np.float32)
    cnt = np.zeros((H,W), dtype=np.float32)
    stride = tile - overlap
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, x0 = max(y,0), max(x,0)
            yy = min(y0+tile, H); xx = min(x0+tile, W)
            ca = before[y0:yy, x0:xx]; cb = after[y0:yy, x0:xx]
            inputs = {"input_before": to_tensor(ca), "input_after": to_tensor(cb)}
            prob = session.run(None, inputs)[0][0,0]
            out[y0:yy, x0:xx] += prob
            cnt[y0:yy, x0:xx] += 1.0
    return out / np.maximum(cnt, 1e-6)

def make_overlay(base, mask, alpha=0.5):
    mask_rgb = np.stack([mask]*3, axis=-1)
    overlay = ((1-alpha)*base + alpha*mask_rgb).astype(np.uint8)
    return overlay

def absdiff_rgb(a,b): return cv2.absdiff(a,b)
def absdiff_gray(a,b): return cv2.cvtColor(absdiff_rgb(a,b), cv2.COLOR_RGB2GRAY)

# -------- Streamlit UI --------
st.title("Satellite Change Detection (Batch ONNX)")

onnx_model_path = st.text_input("Path to ONNX model", "../onnx_model/model.onnx")
if not Path(onnx_model_path).exists():
    st.warning("Model file not found!")
    st.stop()
session = ort.InferenceSession(onnx_model_path)

csv_file = st.file_uploader("Upload CSV (columns: before,after)", type=["csv"])
tile = st.number_input("Tile size", value=512, min_value=0)
overlap = st.number_input("Overlap", value=32, min_value=0)
thr = st.slider("Threshold", 0.0, 1.0, 0.5)
min_blob = st.number_input("Min blob size", 0, 1000, 50)
save_outputs = st.checkbox("Save outputs to folder", value=True)

outdir = st.text_input("Output folder", "outputs") if save_outputs else None
if save_outputs and outdir:
    Path(outdir).mkdir(exist_ok=True)
    mask_dir = Path(outdir)/"masks"; mask_dir.mkdir(exist_ok=True)
    overlay_dir = Path(outdir)/"overlays"; overlay_dir.mkdir(exist_ok=True)
    heatmap_dir = Path(outdir)/"heatmaps"; heatmap_dir.mkdir(exist_ok=True)
    diff_dir = Path(outdir)/"diffs"; diff_dir.mkdir(exist_ok=True)
    diff_gray_dir = Path(outdir)/"diffs_gray"; diff_gray_dir.mkdir(exist_ok=True)
    diff_masked_dir = Path(outdir)/"diffs_masked"; diff_masked_dir.mkdir(exist_ok=True)

if csv_file:
    df = pd.read_csv(csv_file)
    st.write(f"Processing {len(df)} image pairs...")

    for idx, row in df.iterrows():
        try:
            before_path = row['before']; after_path = row['after']
            before = read_image(before_path)
            after  = read_image(after_path)

            prob = predict_onnx(session, before, after, tile=tile, overlap=overlap)
            mask = (prob >= thr).astype(np.uint8)*255

            if min_blob > 0:
                nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                sizes = stats[1:, -1]
                cleaned = np.zeros_like(mask)
                for i, s in enumerate(sizes, start=1):
                    if s >= min_blob:
                        cleaned[output==i] = 255
                mask = cleaned

            overlay = make_overlay(before, mask)
            diff = absdiff_rgb(before, after)
            diff_gray = absdiff_gray(before, after)
            diff_masked = diff * (mask[...,None]>0)

            st.subheader(f"Pair {idx}: {Path(before_path).name}")
            cols = st.columns(6)
            cols[0].image(mask, caption="Mask", width=128)
            cols[1].image(overlay, caption="Overlay", width=128)
            cols[2].image(prob, caption="Probability", width=128)
            cols[3].image(diff, caption="RGB Diff", width=128)
            cols[4].image(diff_gray, caption="Gray Diff", width=128)
            cols[5].image(diff_masked, caption="Masked Diff", width=128)

            if save_outputs:
                base_name = Path(before_path).stem
                cv2.imwrite(str(mask_dir/f"{base_name}.png"), mask)
                cv2.imwrite(str(overlay_dir/f"{base_name}.png"), overlay)
                cv2.imwrite(str(heatmap_dir/f"{base_name}.png"), (prob*255).astype(np.uint8))
                cv2.imwrite(str(diff_dir/f"{base_name}.png"), diff)
                cv2.imwrite(str(diff_gray_dir/f"{base_name}.png"), diff_gray)
                cv2.imwrite(str(diff_masked_dir/f"{base_name}.png"), diff_masked)

        except Exception as e:
            st.error(f"Error processing {row['before']} / {row['after']}: {e}")
