# streamlit_predict_onnx.py
"""
Streamlit app wrapping predict_batch_onnx.py functionality.

Usage:
    pip install -r requirements.txt
    streamlit run streamlit_predict_onnx.py

Notes:
 - The CSV must contain pairs of image paths using columns 'before'/'after' (or 'A'/'B').
 - If your CSV references local filesystem images, upload the CSV only and the app will try to use the paths as-is (the server must have access).
 - If you don't want to rely on filesystem paths, upload a ZIP archive containing the images (preserve filenames). The app will try to match basenames in the CSV to files inside the ZIP.
"""
import os
import io
import csv
import tempfile
import zipfile
from typing import List, Tuple

import streamlit as st
from tqdm import tqdm
import numpy as np
import cv2
import onnxruntime as ort

# -----------------------------
# Utilities (adapted from original script)
# -----------------------------

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def tile_image(img: np.ndarray, tile_size: int, overlap: int) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []
    pos = []
    if step <= 0:
        raise ValueError("tile_size must be > overlap")
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = y
            x1 = x
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tiles.append(img[y1:y2, x1:x2].copy())
            pos.append((y1, y2, x1, x2))
    return tiles, pos


def stitch_tiles(pred_tiles: List[np.ndarray], positions: List[Tuple[int,int,int,int]], full_shape: Tuple[int,int,int]) -> np.ndarray:
    out_h, out_w = full_shape[0], full_shape[1]
    out = np.zeros((out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)
    for tile, (y1, y2, x1, x2) in zip(pred_tiles, positions):
        th, tw = tile.shape[:2]
        out[y1:y1+th, x1:x1+tw] += tile
        count[y1:y1+th, x1:x1+tw] += 1.0
    out = out / np.maximum(count, 1e-6)
    return out

# -----------------------------
# Prediction code (synchronous; updates Streamlit progress)
# -----------------------------

def load_onnx_session(model_path: str):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_batch(csv_rows, model_path: str, outdir: str,
                  overlay_dir: str = None, overlay_on: str = 'before', overlay_alpha: float = 0.5,
                  heatmap_dir: str = None, diff_dir: str = None,
                  tile: int = 512, overlap: int = 32, thr: float = 0.15,
                  progress_callback=None):
    os.makedirs(outdir, exist_ok=True)
    if overlay_dir: os.makedirs(overlay_dir, exist_ok=True)
    if heatmap_dir: os.makedirs(heatmap_dir, exist_ok=True)
    if diff_dir: os.makedirs(diff_dir, exist_ok=True)

    session = load_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total = len(csv_rows)
    for idx, row in enumerate(csv_rows):
        before_path = row.get('before') or row.get('A') or row.get('before_path') or row.get('imgA')
        after_path  = row.get('after')  or row.get('B') or row.get('after_path')  or row.get('imgB')
        if not before_path or not after_path:
            st.warning(f"[skip] CSV row missing before/after: {row}")
            continue

        try:
            before_img = read_image(before_path)
            after_img = read_image(after_path)
        except Exception as e:
            st.error(f"[error] reading images for {before_path}: {e}")
            continue

        if before_img.shape[:2] != after_img.shape[:2]:
            st.warning(f"[skip] shape mismatch: {before_path} {before_img.shape} vs {after_path} {after_img.shape}")
            continue

        h, w = before_img.shape[:2]
        tiles_before, positions = tile_image(before_img, tile, overlap)
        tiles_after, _ = tile_image(after_img, tile, overlap)

        pred_tiles = []
        for tb, ta in zip(tiles_before, tiles_after):
            th, tw = tb.shape[:2]
            pad_h = max(0, tile - th)
            pad_w = max(0, tile - tw)
            if pad_h > 0 or pad_w > 0:
                tb_pad = cv2.copyMakeBorder((tb*255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                ta_pad = cv2.copyMakeBorder((ta*255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                tb_pad = tb_pad.astype(np.float32) / 255.0
                ta_pad = ta_pad.astype(np.float32) / 255.0
            else:
                tb_pad = tb
                ta_pad = ta

            x = np.concatenate([tb_pad, ta_pad], axis=2)
            x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

            try:
                out = session.run([output_name], {input_name: x})[0]
            except Exception as e:
                st.error(f"[error] model run failed on tile: {e}")
                raise

            pred_tile = out[0, 0, :th, :tw].astype(np.float32)
            pred_tiles.append(pred_tile)

        pred_full = stitch_tiles(pred_tiles, positions, before_img.shape)
        pred_mask = (pred_full > thr).astype(np.uint8) * 255

        base = os.path.splitext(os.path.basename(before_path))[0]
        mask_out_path = os.path.join(outdir, f"{base}_mask.png")
        cv2.imwrite(mask_out_path, pred_mask)

        if overlay_dir:
            if overlay_on == 'after':
                base_img = (after_img * 255).astype(np.uint8)
            else:
                base_img = (before_img * 255).astype(np.uint8)
            overlay_color = np.zeros_like(base_img)
            overlay_color[:, :, 0] = pred_mask
            overlayed = cv2.addWeighted(base_img, 1.0 - overlay_alpha, overlay_color, overlay_alpha, 0)
            overlay_out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
            cv2.imwrite(overlay_out_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

        if heatmap_dir:
            heat = cv2.applyColorMap((np.clip(pred_full, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_out_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
            cv2.imwrite(heat_out_path, heat)

        if diff_dir:
            diff_img = cv2.absdiff((before_img * 255).astype(np.uint8), (after_img * 255).astype(np.uint8))
            diff_out = os.path.join(diff_dir, f"{base}_diff.png")
            cv2.imwrite(diff_out, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))

        if progress_callback:
            progress_callback(idx + 1, total)

    return outdir

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.title("Batch ONNX Change-Prediction (before+after)")

    with st.sidebar:
        st.header("Inputs")
        csv_file = st.file_uploader("Upload CSV (columns 'before','after' or 'A','B')", type=['csv'])
        model_file = st.file_uploader("Upload ONNX model (.onnx)", type=['onnx'])
        zip_images = st.file_uploader("(Optional) ZIP of images referenced by CSV (preserve filenames)", type=['zip'])

        st.header("Output options")
        outdir_name = st.text_input("Output subfolder (will be created in a temp dir)", value="predictions")
        overlay_dirname = st.text_input("Overlay folder name (leave empty to skip)", value="overlays")
        heatmap_dirname = st.text_input("Heatmap folder name (leave empty to skip)", value="heatmaps")
        diff_dirname = st.text_input("Diff folder name (leave empty to skip)", value="diffs")

        st.header("Model / Tiling")
        tile = st.number_input("Tile size", value=512, min_value=1)
        overlap = st.number_input("Tile overlap", value=32, min_value=0)
        thr = st.slider("Threshold for mask", min_value=0.0, max_value=1.0, value=0.15)
        overlay_on = st.selectbox("Overlay on", options=["before", "after"], index=1)
        overlay_alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.5)

    run = st.button("Run prediction")

    if run:
        if csv_file is None or model_file is None:
            st.error("Please upload both CSV and ONNX model files.")
            return

        # Prepare temp workspace
        tmp = tempfile.TemporaryDirectory()
        tmpdir = tmp.name
        st.info(f"Working in temporary directory: {tmpdir}")

        # Save model
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, 'wb') as f:
            f.write(model_file.getbuffer())

        # Extract ZIP if provided
        extracted_dir = None
        if zip_images is not None:
            z = zipfile.ZipFile(io.BytesIO(zip_images.getvalue()))
            extracted_dir = os.path.join(tmpdir, "images")
            z.extractall(extracted_dir)

        # Read CSV rows and resolve paths
        csv_text = io.StringIO(csv_file.getvalue().decode('utf-8'))
        reader = csv.DictReader(csv_text)
        rows = []
        unmatched = []
        for r in reader:
            before = r.get('before') or r.get('A') or r.get('before_path') or r.get('imgA')
            after = r.get('after') or r.get('B') or r.get('after_path') or r.get('imgB')
            if extracted_dir:
                # try to find files by basename inside extracted_dir
                def resolve(p):
                    if p is None:
                        return None
                    if os.path.isabs(p) and os.path.exists(p):
                        return p
                    # try basename
                    candidate = os.path.join(extracted_dir, os.path.basename(p))
                    if os.path.exists(candidate):
                        return candidate
                    # else try scanning (expensive for very large zips)
                    for root, _, files in os.walk(extracted_dir):
                        if os.path.basename(p) in files:
                            return os.path.join(root, os.path.basename(p))
                    return p
                before = resolve(before)
                after = resolve(after)
            rows.append({
                'before': before,
                'after': after
            })

        # Create output folders
        out_root = os.path.join(tmpdir, outdir_name)
        overlay_dir = os.path.join(tmpdir, overlay_dirname) if overlay_dirname else None
        heatmap_dir = os.path.join(tmpdir, heatmap_dirname) if heatmap_dirname else None
        diff_dir = os.path.join(tmpdir, diff_dirname) if diff_dirname else None

        progress_text = st.empty()
        progress_bar = st.progress(0)

        def progress_cb(done, total):
            frac = done / float(total)
            progress_bar.progress(min(1.0, frac))
            progress_text.text(f"Processed {done}/{total}")

        try:
            result_dir = predict_batch(rows, model_path, out_root,
                                       overlay_dir=overlay_dir,
                                       overlay_on=overlay_on,
                                       overlay_alpha=overlay_alpha,
                                       heatmap_dir=heatmap_dir,
                                       diff_dir=diff_dir,
                                       tile=tile, overlap=overlap, thr=thr,
                                       progress_callback=progress_cb)
        except Exception as e:
            st.exception(e)
            return

        # Create a ZIP of the output for download
        zip_out_path = os.path.join(tmpdir, "results.zip")
        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(result_dir):
                for fn in files:
                    absf = os.path.join(root, fn)
                    arcname = os.path.relpath(absf, result_dir)
                    zf.write(absf, arcname)
            # also include overlays/heatmaps/diffs if present
            for folder in (overlay_dir, heatmap_dir, diff_dir):
                if folder and os.path.exists(folder):
                    for root, _, files in os.walk(folder):
                        for fn in files:
                            absf = os.path.join(root, fn)
                            arcname = os.path.relpath(absf, tmpdir)
                            zf.write(absf, arcname)

        with open(zip_out_path, 'rb') as f:
            st.download_button("Download results (ZIP)", data=f, file_name="results.zip")

        st.success("Done. Download results using the button above.")

if __name__ == '__main__':
    main()
