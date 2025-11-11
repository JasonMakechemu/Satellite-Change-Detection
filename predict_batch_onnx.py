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
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # Convert BGR -> RGB and to float32 [0..1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def tile_image(img: np.ndarray, tile_size: int, overlap: int) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    """Split image into overlapping tiles. Returns list of tiles (H'xW'xC) and positions (y1,y2,x1,x2)."""
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
    """Stitch tile predictions (float arrays HxW) back into full-size probability map."""
    out_h, out_w = full_shape[0], full_shape[1]
    out = np.zeros((out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)
    for tile, (y1, y2, x1, x2) in zip(pred_tiles, positions):
        th, tw = tile.shape[:2]
        out[y1:y1+th, x1:x1+tw] += tile
        count[y1:y1+th, x1:x1+tw] += 1.0
    # avoid division by zero
    out = out / np.maximum(count, 1e-6)
    return out


# -----------------------------
# Prediction
# -----------------------------
def predict_batch(csv_path: str, model_path: str, outdir: str,
                  overlay_dir: str = None, overlay_on: str = 'before', overlay_alpha: float = 0.5,
                  heatmap_dir: str = None, diff_dir: str = None,
                  tile: int = 512, overlap: int = 32, thr: float = 0.15):
    os.makedirs(outdir, exist_ok=True)
    if overlay_dir: os.makedirs(overlay_dir, exist_ok=True)
    if heatmap_dir: os.makedirs(heatmap_dir, exist_ok=True)
    if diff_dir: os.makedirs(diff_dir, exist_ok=True)

    # Load ONNX model (CPU)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Read CSV rows
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"No rows found in {csv_path}")
        return

    # Process rows with progress bar
    for row in tqdm(rows, desc="predict-batch"):
        # support either 'before'/'after' or 'A'/'B' column naming
        before_path = row.get('before') or row.get('A') or row.get('before_path') or row.get('imgA')
        after_path  = row.get('after')  or row.get('B') or row.get('after_path')  or row.get('imgB')
        if not before_path or not after_path:
            print(f"[skip] CSV row missing before/after: {row}")
            continue

        try:
            before_img = read_image(before_path)
            after_img = read_image(after_path)
        except Exception as e:
            print(f"[error] reading images for {before_path}: {e}")
            continue

        if before_img.shape[:2] != after_img.shape[:2]:
            print(f"[skip] shape mismatch: {before_path} {before_img.shape} vs {after_path} {after_img.shape}")
            continue

        h, w = before_img.shape[:2]

        # Create tiles
        tiles_before, positions = tile_image(before_img, tile, overlap)
        tiles_after, _ = tile_image(after_img, tile, overlap)

        pred_tiles = []
        for tb, ta in zip(tiles_before, tiles_after):
            th, tw = tb.shape[:2]

            # Pad to model tile size if necessary (reflect padding)
            pad_h = max(0, tile - th)
            pad_w = max(0, tile - tw)
            if pad_h > 0 or pad_w > 0:
                tb_pad = cv2.copyMakeBorder((tb*255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                ta_pad = cv2.copyMakeBorder((ta*255).astype(np.uint8), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                # convert back to float32 [0..1]
                tb_pad = tb_pad.astype(np.float32) / 255.0
                ta_pad = ta_pad.astype(np.float32) / 255.0
            else:
                tb_pad = tb
                ta_pad = ta

            # Concatenate channels: (H,W,6) -> (1,6,H,W)
            x = np.concatenate([tb_pad, ta_pad], axis=2)
            x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

            # Run ONNX model
            try:
                out = session.run([output_name], {input_name: x})[0]
            except Exception as e:
                print(f"[error] model run failed on tile: {e}")
                raise

            # out expected shape: (1,1,H,W) or similar; extract and crop to original tile size
            pred_tile = out[0, 0, :th, :tw].astype(np.float32)
            pred_tiles.append(pred_tile)

        # Stitch tile predictions back to full image
        pred_full = stitch_tiles(pred_tiles, positions, before_img.shape)
        pred_mask = (pred_full > thr).astype(np.uint8) * 255

        # filenames
        base = os.path.splitext(os.path.basename(before_path))[0]
        mask_out_path = os.path.join(outdir, f"{base}_mask.png")
        cv2.imwrite(mask_out_path, pred_mask)

        # overlay (on specified image)
        if overlay_dir:
            if overlay_on == 'after':
                base_img = (after_img * 255).astype(np.uint8)
            else:
                base_img = (before_img * 255).astype(np.uint8)

            overlay_color = np.zeros_like(base_img)
            overlay_color[:, :, 0] = pred_mask  # red in RGB image
            overlayed = cv2.addWeighted(base_img, 1.0 - overlay_alpha, overlay_color, overlay_alpha, 0)
            overlay_out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
            cv2.imwrite(overlay_out_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

        # heatmap
        if heatmap_dir:
            heat = cv2.applyColorMap((np.clip(pred_full, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_out_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
            cv2.imwrite(heat_out_path, heat)

        # diff
        if diff_dir:
            diff_img = cv2.absdiff((before_img * 255).astype(np.uint8), (after_img * 255).astype(np.uint8))
            diff_out = os.path.join(diff_dir, f"{base}_diff.png")
            cv2.imwrite(diff_out, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))

    print("Done.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch predict using 6-channel ONNX model (before+after).")
    p.add_argument("--csv", required=True, help="CSV file with columns 'before','after' (or 'A','B').")
    p.add_argument("--model", required=True, help="ONNX model path.")
    p.add_argument("--outdir", required=True, help="Directory to save predicted masks.")
    p.add_argument("--overlay_dir", help="Directory to save overlays (optional).")
    p.add_argument("--overlay_on", choices=["before", "after"], default="after", help="Base image for overlay.")
    p.add_argument("--overlay_alpha", type=float, default=0.5, help="Overlay alpha (0..1).")
    p.add_argument("--heatmap_dir", help="Directory for heatmaps (optional).")
    p.add_argument("--diff_dir", help="Directory for diffs (optional).")
    p.add_argument("--tile", type=int, default=512, help="Tile size (must match model patch size).")
    p.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels.")
    p.add_argument("--thr", type=float, default=0.15, help="Threshold for binary mask.")

    args = p.parse_args()

    predict_batch(csv_path=args.csv,
                  model_path=args.model,
                  outdir=args.outdir,
                  overlay_dir=args.overlay_dir,
                  overlay_on=args.overlay_on,
                  overlay_alpha=args.overlay_alpha,
                  heatmap_dir=args.heatmap_dir,
                  diff_dir=args.diff_dir,
                  tile=args.tile,
                  overlap=args.overlap,
                  thr=args.thr)
