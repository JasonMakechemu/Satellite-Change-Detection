#!/usr/bin/env python3
"""
predict_batch_onnx.py

Predict change masks with an ONNX model that expects a single 6-channel input
(before RGB + after RGB). Handles large images with tiling + padding at edges,
then stitches predictions back together.

Example:
python predict_batch_onnx.py \
  --csv levir_val.csv \
  --model model.onnx \
  --outdir predictions \
  --overlay_dir overlays \
  --heatmap_dir heatmaps \
  --diff_dir diffs \
  --tile 512 \
  --overlap 32 \
  --thr 0.15
"""
import os
import argparse
import csv
import math
from tqdm import tqdm
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# -----------------------------
# Utilities
# -----------------------------

def read_image(path: str) -> np.ndarray:
    """
    Load an image from disk, convert it from BGR (OpenCV default) to RGB,
    and normalise pixel values to the [0, 1] float32 range.
    Raises FileNotFoundError if the path cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # OpenCV loads as BGR; convert to RGB so channel order matches training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def tile_image(
    img: np.ndarray,
    tile_size: int,
    overlap: int,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Divide a large image into overlapping square tiles for inference.

    Overlapping tiles prevent hard seam artefacts at tile boundaries: each
    boundary region is covered by at least two tiles, and predictions are
    averaged during stitching.

    Args:
        img:       Input image array of shape (H, W, C).
        tile_size: Height/width of each tile in pixels.
        overlap:   Number of pixels each tile shares with its neighbour.

    Returns:
        tiles: List of image crops, each up to (tile_size x tile_size x C).
               Edge tiles may be smaller when the image doesn't divide evenly.
        pos:   Parallel list of (y1, y2, x1, x2) pixel coordinates in the
               original image that correspond to each tile.
    """
    h, w = img.shape[:2]
    step = tile_size - overlap  # how far to advance the window each step
    tiles = []
    pos = []

    if step <= 0:
        raise ValueError("tile_size must be > overlap")

    # Slide window in row-major order; clamp at image edges
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = y
            x1 = x
            y2 = min(y + tile_size, h)  # clamp so we don't read past the bottom
            x2 = min(x + tile_size, w)  # clamp so we don't read past the right
            tiles.append(img[y1:y2, x1:x2].copy())
            pos.append((y1, y2, x1, x2))

    return tiles, pos


def stitch_tiles(
    pred_tiles: List[np.ndarray],
    positions: List[Tuple[int, int, int, int]],
    full_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct a full-resolution probability map by averaging overlapping
    tile predictions back into a canvas.

    Where tiles overlap, their softmax/sigmoid scores are summed and then
    divided by the number of tiles that covered each pixel, so every pixel
    receives a proper average rather than a biased sum.

    Args:
        pred_tiles: List of per-tile probability arrays, each shape (H', W').
        positions:  Matching list of (y1, y2, x1, x2) source coordinates.
        full_shape: (H, W, C) shape of the original image; only H and W are used.

    Returns:
        Averaged probability map of shape (H, W), dtype float32.
    """
    out_h, out_w = full_shape[0], full_shape[1]

    # Accumulator for summed predictions and per-pixel tile counts
    out   = np.zeros((out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)

    for tile, (y1, y2, x1, x2) in zip(pred_tiles, positions):
        th, tw = tile.shape[:2]
        # Add tile prediction into the canvas at the correct position
        out  [y1:y1+th, x1:x1+tw] += tile
        count[y1:y1+th, x1:x1+tw] += 1.0

    # Divide by count to turn sums into averages; guard against zero-count pixels
    out = out / np.maximum(count, 1e-6)
    return out


# -----------------------------
# Prediction
# -----------------------------

def predict_batch(
    csv_path: str,
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
):
    """
    Run change-detection inference on every image pair listed in a CSV file.

    For each pair the function:
      1. Tiles both images into overlapping patches.
      2. Pads edge patches to the required model input size.
      3. Stacks before/after patches into a 6-channel tensor and runs ONNX inference.
      4. Stitches per-tile probability maps back into a full-resolution map.
      5. Thresholds the map to produce a binary mask and writes it to disk.
      6. Optionally writes overlay, heatmap, and pixel-difference images.

    Args:
        csv_path:      Path to a CSV with 'before'/'after' (or 'A'/'B') columns.
        model_path:    Path to the ONNX model file.
        outdir:        Output directory for binary masks.
        overlay_dir:   Output directory for coloured overlays (None = skip).
        overlay_on:    Which image ('before' or 'after') to draw the overlay on.
        overlay_alpha: Blend factor for the red change overlay (0 = invisible, 1 = opaque).
        heatmap_dir:   Output directory for JET-coloured probability heatmaps (None = skip).
        diff_dir:      Output directory for per-pixel absolute-difference images (None = skip).
        tile:          Tile size in pixels; must match the model's expected input resolution.
        overlap:       Overlap between adjacent tiles in pixels.
        thr:           Probability threshold above which a pixel is marked as changed.
    """
    # Create all required output directories up-front
    os.makedirs(outdir, exist_ok=True)
    if overlay_dir: os.makedirs(overlay_dir, exist_ok=True)
    if heatmap_dir: os.makedirs(heatmap_dir, exist_ok=True)
    if diff_dir:    os.makedirs(diff_dir,    exist_ok=True)

    # Initialise the ONNX runtime session on CPU
    session     = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs() [0].name  # name of the single input node
    output_name = session.get_outputs()[0].name  # name of the primary output node

    # Parse every row from the CSV into memory
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"No rows found in {csv_path}")
        return

    # Iterate over image pairs with a console progress bar
    for row in tqdm(rows, desc="predict-batch"):
        # Support multiple column-naming conventions (before/after, A/B, etc.)
        before_path = row.get('before') or row.get('A') or row.get('before_path') or row.get('imgA')
        after_path  = row.get('after')  or row.get('B') or row.get('after_path')  or row.get('imgB')

        if not before_path or not after_path:
            print(f"[skip] CSV row missing before/after: {row}")
            continue

        # Load both images; skip the pair on any read error
        try:
            before_img = read_image(before_path)
            after_img  = read_image(after_path)
        except Exception as e:
            print(f"[error] reading images for {before_path}: {e}")
            continue

        # Both images must have the same spatial dimensions for change detection
        if before_img.shape[:2] != after_img.shape[:2]:
            print(f"[skip] shape mismatch: {before_path} {before_img.shape} vs {after_path} {after_img.shape}")
            continue

        h, w = before_img.shape[:2]

        # Tile both images using identical positions so patches align spatially
        tiles_before, positions = tile_image(before_img, tile, overlap)
        tiles_after,  _         = tile_image(after_img,  tile, overlap)

        pred_tiles = []  # will hold per-tile probability arrays

        for tb, ta in zip(tiles_before, tiles_after):
            th, tw = tb.shape[:2]  # actual (possibly smaller) tile dimensions

            # --- Padding ---
            # Edge tiles that are smaller than `tile` must be padded to the
            # exact size the model expects. Reflect padding avoids introducing
            # artificial black borders that could bias predictions.
            pad_h = max(0, tile - th)
            pad_w = max(0, tile - tw)

            if pad_h > 0 or pad_w > 0:
                # cv2.copyMakeBorder expects uint8; convert, pad, then restore float32
                tb_pad = cv2.copyMakeBorder(
                    (tb * 255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )
                ta_pad = cv2.copyMakeBorder(
                    (ta * 255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )
                tb_pad = tb_pad.astype(np.float32) / 255.0
                ta_pad = ta_pad.astype(np.float32) / 255.0
            else:
                tb_pad = tb
                ta_pad = ta

            # --- Build model input tensor ---
            # Concatenate along the channel axis: (H, W, 3) + (H, W, 3) -> (H, W, 6)
            x = np.concatenate([tb_pad, ta_pad], axis=2)
            # Reorder to NCHW format expected by most PyTorch-exported ONNX models:
            # (H, W, 6) -> (6, H, W) -> (1, 6, H, W)
            x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

            # --- ONNX inference ---
            try:
                out = session.run([output_name], {input_name: x})[0]
            except Exception as e:
                print(f"[error] model run failed on tile: {e}")
                raise

            # Model output is expected to be (1, 1, H, W); extract the spatial map
            # and crop away any padding that was added to the edge tile
            pred_tile = out[0, 0, :th, :tw].astype(np.float32)
            pred_tiles.append(pred_tile)

        # --- Stitch & threshold ---
        # Average overlapping tile predictions back into a full-resolution map
        pred_full = stitch_tiles(pred_tiles, positions, before_img.shape)
        # Apply threshold to get a binary mask: 255 = changed, 0 = unchanged
        pred_mask = (pred_full > thr).astype(np.uint8) * 255

        # Derive a base filename from the 'before' image path
        base = os.path.splitext(os.path.basename(before_path))[0]

        # --- Save binary mask ---
        mask_out_path = os.path.join(outdir, f"{base}_mask.png")
        cv2.imwrite(mask_out_path, pred_mask)

        # --- Optional: colour overlay of change mask on source image ---
        if overlay_dir:
            # Choose which image to draw on
            if overlay_on == 'after':
                base_img = (after_img  * 255).astype(np.uint8)
            else:
                base_img = (before_img * 255).astype(np.uint8)

            # Create a solid red image where the mask fires, then alpha-blend
            overlay_color = np.zeros_like(base_img)
            overlay_color[:, :, 0] = pred_mask  # red channel in RGB space

            overlayed = cv2.addWeighted(
                base_img,      1.0 - overlay_alpha,  # original image contribution
                overlay_color, overlay_alpha,         # red mask contribution
                0,
            )

            overlay_out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
            # Convert back to BGR before writing (OpenCV convention)
            cv2.imwrite(overlay_out_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

        # --- Optional: JET heatmap of raw change probabilities ---
        if heatmap_dir:
            # Scale probabilities to uint8 and apply a perceptually intuitive colormap
            heat = cv2.applyColorMap(
                (np.clip(pred_full, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heat_out_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
            cv2.imwrite(heat_out_path, heat)

        # --- Optional: absolute pixel-difference image between before and after ---
        if diff_dir:
            diff_img = cv2.absdiff(
                (before_img * 255).astype(np.uint8),
                (after_img  * 255).astype(np.uint8),
            )
            diff_out = os.path.join(diff_dir, f"{base}_diff.png")
            # Convert RGB -> BGR for OpenCV's imwrite
            cv2.imwrite(diff_out, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))

    print("Done.")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Batch predict using 6-channel ONNX model (before+after)."
    )
    p.add_argument("--csv",           required=True,  help="CSV file with columns 'before','after' (or 'A','B').")
    p.add_argument("--model",         required=True,  help="ONNX model path.")
    p.add_argument("--outdir",        required=True,  help="Directory to save predicted masks.")
    p.add_argument("--overlay_dir",                   help="Directory to save overlays (optional).")
    p.add_argument("--overlay_on",    choices=["before", "after"], default="after",
                                                       help="Base image for overlay.")
    p.add_argument("--overlay_alpha", type=float, default=0.5,
                                                       help="Overlay alpha (0..1).")
    p.add_argument("--heatmap_dir",                   help="Directory for heatmaps (optional).")
    p.add_argument("--diff_dir",                      help="Directory for diffs (optional).")
    p.add_argument("--tile",          type=int,   default=512,
                                                       help="Tile size (must match model patch size).")
    p.add_argument("--overlap",       type=int,   default=32,
                                                       help="Tile overlap in pixels.")
    p.add_argument("--thr",           type=float, default=0.15,
                                                       help="Threshold for binary mask.")

    args = p.parse_args()

    predict_batch(
        csv_path    = args.csv,
        model_path  = args.model,
        outdir      = args.outdir,
        overlay_dir = args.overlay_dir,
        overlay_on  = args.overlay_on,
        overlay_alpha = args.overlay_alpha,
        heatmap_dir = args.heatmap_dir,
        diff_dir    = args.diff_dir,
        tile        = args.tile,
        overlap     = args.overlap,
        thr         = args.thr,
    )
