# streamlit_predict_onnx.py
"""
Streamlit app wrapping predict_batch_onnx.py functionality.

Usage:
    pip install -r requirements.txt
    streamlit run streamlit_predict_onnx.py

Notes:
 - The CSV must contain pairs of image paths using columns 'before'/'after' (or 'A'/'B').
 - If your CSV references local filesystem images, upload the CSV only and the app will try
   to use the paths as-is (the server must have access to those paths).
 - If you don't want to rely on filesystem paths, upload a ZIP archive containing the images
   (preserve filenames). The app will try to match basenames in the CSV to files inside the ZIP.
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
    """
    Load an image from disk, convert BGR -> RGB, and normalise to float32 [0, 1].
    Raises FileNotFoundError if the path cannot be opened by OpenCV.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # OpenCV loads images in BGR order; convert to RGB to match training convention
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def tile_image(
    img: np.ndarray,
    tile_size: int,
    overlap: int,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Divide a large image into overlapping square tiles.

    Overlapping tiles prevent hard seam artefacts: the boundary region between
    two adjacent tiles is covered by both tiles, and predictions are averaged
    during stitching, smoothing out any discontinuity.

    Args:
        img:       Input array of shape (H, W, C).
        tile_size: Desired tile height/width in pixels.
        overlap:   Number of pixels shared between neighbouring tiles.

    Returns:
        tiles: List of image crops (may be smaller than tile_size at edges).
        pos:   Parallel list of (y1, y2, x1, x2) coordinates in the original image.
    """
    h, w = img.shape[:2]
    step = tile_size - overlap  # distance between successive tile origins
    tiles = []
    pos = []

    if step <= 0:
        raise ValueError("tile_size must be > overlap")

    # Slide window across the image in row-major order; clamp at boundaries
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = y, x
            y2 = min(y + tile_size, h)  # clamp bottom edge
            x2 = min(x + tile_size, w)  # clamp right edge
            tiles.append(img[y1:y2, x1:x2].copy())
            pos.append((y1, y2, x1, x2))

    return tiles, pos


def stitch_tiles(
    pred_tiles: List[np.ndarray],
    positions: List[Tuple[int, int, int, int]],
    full_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct a full-resolution probability map by averaging overlapping tile
    predictions back into a single canvas.

    Each pixel's final value is the mean of all tile predictions that covered it,
    which smooths boundary artefacts introduced by tiling.

    Args:
        pred_tiles: Per-tile probability arrays, each shape (H', W').
        positions:  Matching (y1, y2, x1, x2) source coordinates for each tile.
        full_shape: (H, W, C) of the original image; only H and W are used.

    Returns:
        Averaged probability map of shape (H, W), dtype float32.
    """
    out_h, out_w = full_shape[0], full_shape[1]

    # Accumulators: sum of predictions and count of tiles per pixel
    out   = np.zeros((out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)

    for tile, (y1, y2, x1, x2) in zip(pred_tiles, positions):
        th, tw = tile.shape[:2]
        out  [y1:y1+th, x1:x1+tw] += tile
        count[y1:y1+th, x1:x1+tw] += 1.0

    # Divide sums by counts; 1e-6 guard prevents division by zero on uncovered pixels
    out = out / np.maximum(count, 1e-6)
    return out


# -----------------------------
# Prediction code (synchronous; updates Streamlit progress bar)
# -----------------------------

def load_onnx_session(model_path: str):
    """
    Create and return an ONNX Runtime inference session running on CPU.
    Separating session creation makes it easy to cache or reuse the session
    across multiple calls without reloading the model from disk.
    """
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_batch(
    csv_rows,
    model_path: str,
    outdir: str,
    overlay_dir: str = None,
    overlay_on: str = 'before',
    overlay_alpha: float = 0.5,
    heatmap_dir: str = None,
    diff_dir: str = None,
    tile: int = 512,
    overlap: int = 32,
    thr: float = 0.15,
    progress_callback=None,
):
    """
    Run change-detection inference on every image pair in csv_rows and write
    results to disk. Optionally calls a progress callback so the Streamlit UI
    can update its progress bar in real time.

    Processing pipeline per pair:
      1. Tile both images into overlapping patches.
      2. Pad edge patches to the model's expected tile size.
      3. Stack before+after into a 6-channel tensor and run ONNX inference.
      4. Stitch per-tile probability maps back into a full-resolution map.
      5. Threshold to produce a binary mask and save it.
      6. Optionally produce overlays, heatmaps, and diff images.

    Args:
        csv_rows:          List of dicts with 'before' and 'after' path keys.
        model_path:        Path to the ONNX model file on disk.
        outdir:            Directory where binary masks are saved.
        overlay_dir:       Directory for coloured change overlays (None = skip).
        overlay_on:        Which image ('before'|'after') to draw the overlay on.
        overlay_alpha:     Blend strength of the red change overlay (0..1).
        heatmap_dir:       Directory for JET-coloured probability heatmaps (None = skip).
        diff_dir:          Directory for absolute pixel-difference images (None = skip).
        tile:              Tile size in pixels; must match model's input resolution.
        overlap:           Overlap between adjacent tiles in pixels.
        thr:               Probability threshold above which a pixel is flagged as changed.
        progress_callback: Optional callable(done: int, total: int) for UI progress updates.

    Returns:
        Path to outdir (the directory containing binary masks).
    """
    # Create all required output directories up-front so later writes never fail
    os.makedirs(outdir, exist_ok=True)
    if overlay_dir: os.makedirs(overlay_dir, exist_ok=True)
    if heatmap_dir: os.makedirs(heatmap_dir, exist_ok=True)
    if diff_dir:    os.makedirs(diff_dir,    exist_ok=True)

    # Load the ONNX model and look up the I/O node names
    session     = load_onnx_session(model_path)
    input_name  = session.get_inputs() [0].name
    output_name = session.get_outputs()[0].name

    total = len(csv_rows)

    for idx, row in enumerate(csv_rows):
        # Support multiple column-naming conventions for the image pair paths
        before_path = row.get('before') or row.get('A') or row.get('before_path') or row.get('imgA')
        after_path  = row.get('after')  or row.get('B') or row.get('after_path')  or row.get('imgB')

        if not before_path or not after_path:
            st.warning(f"[skip] CSV row missing before/after: {row}")
            continue

        # Load images; surface errors as Streamlit messages rather than crashing
        try:
            before_img = read_image(before_path)
            after_img  = read_image(after_path)
        except Exception as e:
            st.error(f"[error] reading images for {before_path}: {e}")
            continue

        # Change detection requires both images to be the same size
        if before_img.shape[:2] != after_img.shape[:2]:
            st.warning(
                f"[skip] shape mismatch: {before_path} {before_img.shape} "
                f"vs {after_path} {after_img.shape}"
            )
            continue

        h, w = before_img.shape[:2]

        # Tile both images with identical window positions so patches align spatially
        tiles_before, positions = tile_image(before_img, tile, overlap)
        tiles_after,  _         = tile_image(after_img,  tile, overlap)

        pred_tiles = []  # accumulate per-tile probability arrays

        for tb, ta in zip(tiles_before, tiles_after):
            th, tw = tb.shape[:2]  # actual tile size (may be smaller at image edges)

            # --- Padding ---
            # Edge tiles smaller than `tile` must be padded so the model receives
            # a consistently sized input. Reflect padding avoids black-border bias.
            pad_h = max(0, tile - th)
            pad_w = max(0, tile - tw)

            if pad_h > 0 or pad_w > 0:
                # cv2.copyMakeBorder requires uint8; convert, pad, then restore float32
                tb_pad = cv2.copyMakeBorder(
                    (tb * 255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )
                ta_pad = cv2.copyMakeBorder(
                    (ta * 255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )
                tb_pad = tb_pad.astype(np.float32) / 255.0
                ta_pad = ta_pad.astype(np.float32) / 255.0
            else:
                tb_pad, ta_pad = tb, ta

            # --- Build model input tensor ---
            # Concatenate before and after along channels: (H,W,3)+(H,W,3) -> (H,W,6)
            # Then reorder to NCHW: (H,W,6) -> (6,H,W) -> (1,6,H,W)
            x = np.concatenate([tb_pad, ta_pad], axis=2)
            x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

            # --- ONNX inference ---
            try:
                out = session.run([output_name], {input_name: x})[0]
            except Exception as e:
                st.error(f"[error] model run failed on tile: {e}")
                raise

            # Extract spatial probability map from (1,1,H,W) and crop padding
            pred_tile = out[0, 0, :th, :tw].astype(np.float32)
            pred_tiles.append(pred_tile)

        # --- Stitch & threshold ---
        pred_full = stitch_tiles(pred_tiles, positions, before_img.shape)
        # Pixels above the threshold are marked as changed (255); others as 0
        pred_mask = (pred_full > thr).astype(np.uint8) * 255

        # Derive a clean base filename from the 'before' image path
        base = os.path.splitext(os.path.basename(before_path))[0]

        # --- Save binary mask ---
        mask_out_path = os.path.join(outdir, f"{base}_mask.png")
        cv2.imwrite(mask_out_path, pred_mask)

        # --- Optional: red change-mask overlay on the chosen source image ---
        if overlay_dir:
            base_img = (
                (after_img  * 255).astype(np.uint8) if overlay_on == 'after'
                else (before_img * 255).astype(np.uint8)
            )
            # Build a red image (in RGB space) where the mask fires
            overlay_color = np.zeros_like(base_img)
            overlay_color[:, :, 0] = pred_mask  # red channel

            overlayed = cv2.addWeighted(
                base_img,      1.0 - overlay_alpha,
                overlay_color, overlay_alpha,
                0,
            )
            overlay_out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
            # Convert RGB -> BGR before writing with OpenCV
            cv2.imwrite(overlay_out_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

        # --- Optional: JET heatmap of raw change probabilities ---
        if heatmap_dir:
            heat = cv2.applyColorMap(
                (np.clip(pred_full, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heat_out_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
            cv2.imwrite(heat_out_path, heat)

        # --- Optional: per-pixel absolute difference between before and after ---
        if diff_dir:
            diff_img = cv2.absdiff(
                (before_img * 255).astype(np.uint8),
                (after_img  * 255).astype(np.uint8),
            )
            diff_out = os.path.join(diff_dir, f"{base}_diff.png")
            cv2.imwrite(diff_out, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))

        # Notify the caller (Streamlit UI) so the progress bar can advance
        if progress_callback:
            progress_callback(idx + 1, total)

    return outdir


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.title("Batch ONNX Change-Prediction (before+after)")

    # ---- Sidebar: all user-facing inputs ----
    with st.sidebar:
        st.header("Inputs")

        # CSV listing image pairs; required to run inference
        csv_file = st.file_uploader(
            "Upload CSV (columns 'before','after' or 'A','B')", type=['csv']
        )
        # ONNX model weights; required to run inference
        model_file = st.file_uploader("Upload ONNX model (.onnx)", type=['onnx'])
        # Optional ZIP archive; needed when the CSV uses relative/bare filenames
        # rather than absolute paths accessible from the server
        zip_images = st.file_uploader(
            "(Optional) ZIP of images referenced by CSV (preserve filenames)",
            type=['zip'],
        )

        st.header("Output options")
        # Sub-folder names inside the temp workspace; empty string disables that output
        outdir_name     = st.text_input("Output subfolder (will be created in a temp dir)", value="predictions")
        overlay_dirname = st.text_input("Overlay folder name (leave empty to skip)",  value="overlays")
        heatmap_dirname = st.text_input("Heatmap folder name (leave empty to skip)",  value="heatmaps")
        diff_dirname    = st.text_input("Diff folder name (leave empty to skip)",      value="diffs")

        st.header("Model / Tiling")
        tile          = st.number_input("Tile size",    value=512, min_value=1)
        overlap       = st.number_input("Tile overlap", value=32,  min_value=0)
        thr           = st.slider("Threshold for mask", min_value=0.0, max_value=1.0, value=0.15)
        overlay_on    = st.selectbox("Overlay on", options=["before", "after"], index=1)
        overlay_alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.5)

    run = st.button("Run prediction")

    if run:
        # Guard: both files are mandatory; surface a clear error rather than crashing
        if csv_file is None or model_file is None:
            st.error("Please upload both CSV and ONNX model files.")
            return

        # Use a managed temp directory so all intermediate files are cleaned up
        # automatically when the request completes (or the app restarts)
        tmp    = tempfile.TemporaryDirectory()
        tmpdir = tmp.name
        st.info(f"Working in temporary directory: {tmpdir}")

        # Write the uploaded ONNX model to disk so ONNX Runtime can load it by path
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, 'wb') as f:
            f.write(model_file.getbuffer())

        # --- Resolve image paths from optional ZIP archive ---
        extracted_dir = None
        if zip_images is not None:
            # Extract the entire archive; filenames from the CSV will be matched
            # against files inside this directory by basename (see resolve() below)
            z = zipfile.ZipFile(io.BytesIO(zip_images.getvalue()))
            extracted_dir = os.path.join(tmpdir, "images")
            z.extractall(extracted_dir)

        # --- Parse CSV and resolve image paths ---
        csv_text = io.StringIO(csv_file.getvalue().decode('utf-8'))
        reader   = csv.DictReader(csv_text)
        rows     = []
        unmatched = []  # reserved for future warnings about unresolvable paths

        for r in reader:
            # Read the path from whichever column convention the CSV uses
            before = r.get('before') or r.get('A') or r.get('before_path') or r.get('imgA')
            after  = r.get('after')  or r.get('B') or r.get('after_path')  or r.get('imgB')

            if extracted_dir:
                def resolve(p):
                    """
                    Try to map a CSV path to an actual file inside the extracted ZIP.

                    Strategy (in order of preference):
                      1. Absolute path that already exists on disk — use as-is.
                      2. Basename match directly under extracted_dir — fast common case.
                      3. Recursive walk of extracted_dir — handles nested sub-folders.
                      4. Return the original path unchanged — will fail at read time
                         with a clear error message.
                    """
                    if p is None:
                        return None
                    # If the CSV embeds absolute server paths and they exist, trust them
                    if os.path.isabs(p) and os.path.exists(p):
                        return p
                    # Flat match: most ZIPs store images at the top level
                    candidate = os.path.join(extracted_dir, os.path.basename(p))
                    if os.path.exists(candidate):
                        return candidate
                    # Deep match: walk sub-directories (slower, used as a fallback)
                    for root, _, files in os.walk(extracted_dir):
                        if os.path.basename(p) in files:
                            return os.path.join(root, os.path.basename(p))
                    # No match found; return original so a useful error surfaces later
                    return p

                before = resolve(before)
                after  = resolve(after)

            rows.append({'before': before, 'after': after})

        # --- Set up output directories inside the temp workspace ---
        out_root   = os.path.join(tmpdir, outdir_name)
        # An empty dirname string means the user opted out of that output type
        overlay_dir = os.path.join(tmpdir, overlay_dirname) if overlay_dirname else None
        heatmap_dir = os.path.join(tmpdir, heatmap_dirname) if heatmap_dirname else None
        diff_dir    = os.path.join(tmpdir, diff_dirname)    if diff_dirname    else None

        # --- Progress indicators ---
        progress_text = st.empty()    # placeholder for "X/Y" text
        progress_bar  = st.progress(0)

        def progress_cb(done: int, total: int):
            """Update the Streamlit progress bar and status text after each image pair."""
            frac = done / float(total)
            progress_bar.progress(min(1.0, frac))
            progress_text.text(f"Processed {done}/{total}")

        # --- Run inference ---
        try:
            result_dir = predict_batch(
                rows, model_path, out_root,
                overlay_dir   = overlay_dir,
                overlay_on    = overlay_on,
                overlay_alpha = overlay_alpha,
                heatmap_dir   = heatmap_dir,
                diff_dir      = diff_dir,
                tile          = tile,
                overlap       = overlap,
                thr           = thr,
                progress_callback = progress_cb,
            )
        except Exception as e:
            # Show full traceback inside the app for easier debugging
            st.exception(e)
            return

        # --- Package all results into a single ZIP for download ---
        # This is the only file the user receives; no temp paths are exposed.
        zip_out_path = os.path.join(tmpdir, "results.zip")

        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Binary masks (always present)
            for root, _, files in os.walk(result_dir):
                for fn in files:
                    absf    = os.path.join(root, fn)
                    arcname = os.path.relpath(absf, result_dir)
                    zf.write(absf, arcname)

            # Optional output folders; each gets its own sub-directory in the ZIP
            for folder in (overlay_dir, heatmap_dir, diff_dir):
                if folder and os.path.exists(folder):
                    for root, _, files in os.walk(folder):
                        for fn in files:
                            absf    = os.path.join(root, fn)
                            # arcname is relative to tmpdir so folder names are preserved
                            arcname = os.path.relpath(absf, tmpdir)
                            zf.write(absf, arcname)

        # Offer the ZIP as a one-click download button
        with open(zip_out_path, 'rb') as f:
            st.download_button("Download results (ZIP)", data=f, file_name="results.zip")

        st.success("Done. Download results using the button above.")


if __name__ == '__main__':
    main()
