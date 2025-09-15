#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 19:20:48 2025

@author: jason
"""

"""
Satellite Change Detection — One-File Train / Predict / Evaluate (CWD-friendly)

This script now assumes **everything lives in the current working directory (CWD)** and supports
**difference images** and **mask overlays** out of the box.

Quickstart (LEVIR‑CD in CWD):
  CWD/
    train/ A/ B/ label/
    val/   A/ B/ label/   (optional)
    test/  A/ B/ label/   (optional)

1) Install deps
   pip install torch torchvision opencv-python pillow numpy tqdm scikit-image

2) Create CSVs (writes levir_train.csv, levir_val.csv, levir_test.csv into CWD)
   python satellite_change_detection.py make-csvs

3) Train (defaults to levir_train.csv in CWD; writes model.pt)
   python satellite_change_detection.py train

4) Evaluate on val/test split (defaults to levir_val.csv, model.pt)
   python satellite_change_detection.py eval

5) Predict a single pair + visualizations
   python satellite_change_detection.py predict \
     --before val/A/1234.png --after val/B/1234.png \
     --out pred_1234.png --thr 0.5 --min-blob 50 \
     --overlay overlay_1234.png --overlay-on before --overlay-alpha 0.5 --outline \
     --diff diff_rgb_1234.png --diff-gray diff_gray_1234.png --diff-masked diff_masked_1234.png \
     --heatmap heat_1234.png

6) **Batch predict** all pairs listed in a CSV into an output folder (with optional viz folders)
   python satellite_change_detection.py predict-batch \
     --csv levir_val.csv --outdir preds_val --thr 0.5 --tile 512 --overlap 32 --min-blob 50 \
     --overlay-dir overlays_val --overlay-on before --overlay-alpha 0.5 --outline \
     --diff-dir diffs_val --diff-gray-dir diffs_gray_val --diff-masked-dir diffs_masked_val \
     --heatmap-dir heats_val

Notes
- Images within a pair must be pixel-aligned and the same size. LEVIR‑CD already is.
- For large custom images, use: --tile 512 --overlap 32
- For slight misregistration on your own data, add: --align
"""

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Model: Siamese U-Net (abs-diff fusion)
# -----------------------------

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = conv_block(cin, cout)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin//2, 2, stride=2)
        self.conv = conv_block(cin, cout)
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))

class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.inc  = conv_block(in_ch, base)
        self.d1   = Down(base, base*2)
        self.d2   = Down(base*2, base*4)
        self.d3   = Down(base*4, base*8)
        self.d4   = Down(base*8, base*16)
        self.u1 = Up(base*16, base*8)
        self.u2 = Up(base*8,  base*4)
        self.u3 = Up(base*4,  base*2)
        self.u4 = Up(base*2,  base)
        self.outc = nn.Conv2d(base, 1, 1)
    def enc(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        return [x1,x2,x3,x4,x5]
    def forward(self, a, b):
        fa = self.enc(a); fb = self.enc(b)
        diffs = [torch.abs(x-y) for x,y in zip(fa, fb)]
        x = self.u1(diffs[4], diffs[3])
        x = self.u2(x,          diffs[2])
        x = self.u3(x,          diffs[1])
        x = self.u4(x,          diffs[0])
        return self.outc(x)

# -----------------------------
# Data loading
# -----------------------------

def read_image(path: str, in_ch: int) -> np.ndarray:
    img = Image.open(path)
    if in_ch == 3:
        img = img.convert('RGB')
    else:
        img = np.array(img)
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[2] != in_ch:
            raise ValueError(f"Image {path} has {img.shape[2]} channels, expected {in_ch}")
        return img
    return np.array(img)


def normalize(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def aug_pair(a: np.ndarray, b: np.ndarray, m: Optional[np.ndarray]):
    if np.random.rand() < 0.5:
        a = np.flip(a, axis=1); b = np.flip(b, axis=1)
        if m is not None: m = np.flip(m, axis=1)
    if np.random.rand() < 0.5:
        a = np.flip(a, axis=0); b = np.flip(b, axis=0)
        if m is not None: m = np.flip(m, axis=0)
    if np.random.rand() < 0.5:
        a = np.rot90(a, 1, axes=(0,1)); b = np.rot90(b, 1, axes=(0,1))
        if m is not None: m = np.rot90(m, 1, axes=(0,1))
    return a.copy(), b.copy(), None if m is None else m.copy()


@dataclass
class Item:
    before: str
    after: str
    mask: Optional[str]

class PairDataset(Dataset):
    def __init__(self, items: List[Item], in_ch: int, tile: int = 256, augment: bool = True):
        self.items = items
        self.in_ch = in_ch
        self.tile = tile
        self.augment = augment
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        a = read_image(it.before, self.in_ch)
        b = read_image(it.after, self.in_ch)
        if a.shape != b.shape:
            raise ValueError(f"Mismatched shapes: {it.before} {a.shape} vs {it.after} {b.shape}")
        m = None
        if it.mask:
            m = Image.open(it.mask)
            m = np.array(m)
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 0).astype(np.float32)
        H, W = a.shape[:2]
        th, tw = self.tile, self.tile
        if H < th or W < tw:
            pad_h = max(0, th - H); pad_w = max(0, tw - W)
            a = np.pad(a, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
            b = np.pad(b, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
            if m is not None: m = np.pad(m, ((0,pad_h),(0,pad_w)), mode='reflect')
            H, W = a.shape[:2]
        y = np.random.randint(0, H - th + 1)
        x = np.random.randint(0, W - tw + 1)
        a = a[y:y+th, x:x+tw]
        b = b[y:y+th, x:x+tw]
        if m is not None: m = m[y:y+th, x:x+tw]
        if self.augment:
            a, b, m = aug_pair(a, b, m)
        a = normalize(a); b = normalize(b)
        a = torch.from_numpy(a.transpose(2,0,1)).float()
        b = torch.from_numpy(b.transpose(2,0,1)).float()
        if m is None:
            m = torch.zeros(1, self.tile, self.tile)
        else:
            m = torch.from_numpy(m[None, ...]).float()
        return a, b, m

# -----------------------------
# Losses & metrics
# -----------------------------

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = (2*(probs*targets)).sum(dim=[2,3])
    den = (probs+targets).sum(dim=[2,3]) + eps
    return 1 - (num/den).mean()

# -----------------------------
# Alignment util (ECC homography on grayscale)
# -----------------------------

def align_image(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    ref_gray = cv2.cvtColor((ref*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if ref.shape[2] >= 3 else (ref[...,0]*255).astype(np.uint8)
    mov_gray = cv2.cvtColor((mov*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if mov.shape[2] >= 3 else (mov[...,0]*255).astype(np.uint8)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(mov, warp_matrix, (mov.shape[1], mov.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned
    except cv2.error:
        return mov

# -----------------------------
# Training helpers
# -----------------------------

def read_items_from_csv(csv_path: str) -> List[Item]:
    items = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(Item(row['before'], row['after'], row.get('mask') or None))
    return items


def split_items(items: List[Item], val_frac: float = 0.2, seed: int = 42) -> Tuple[List[Item], List[Item]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = max(1, int(len(items)*val_frac))
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    return train, val


def train_cmd(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    items = read_items_from_csv(args.csv)
    train_items, val_items = split_items(items, args.val_frac) if args.val_frac > 0 else (items, [])

    train_ds = PairDataset(train_items, in_ch=args.in_ch, tile=args.tile, augment=True)
    val_ds   = PairDataset(val_items,   in_ch=args.in_ch, tile=args.tile, augment=False) if val_items else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False) if val_ds else None

    model = SiameseUNet(in_ch=args.in_ch, base=args.base)
    model.to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs,1))

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for a,b,m in pbar:
            a,b,m = a.to(device), b.to(device), m.to(device)
            logits = model(a,b)
            loss = args.alpha*bce(logits, m) + (1-args.alpha)*dice_loss(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        # validation
        if val_loader:
            model.eval(); vloss = []
            with torch.no_grad():
                for a,b,m in val_loader:
                    a,b,m = a.to(device), b.to(device), m.to(device)
                    logits = model(a,b)
                    loss = args.alpha*bce(logits, m) + (1-args.alpha)*dice_loss(logits, m)
                    vloss.append(loss.item())
            vmean = float(np.mean(vloss)) if vloss else 0.0
            print(f"epoch {epoch} val_loss {vmean:.4f}")
            if vmean < best_val:
                best_val = vmean
                torch.save({'model': model.state_dict(), 'in_ch': args.in_ch, 'base': args.base}, args.out)
                print(f"saved best to {args.out}")
        else:
            # no val set: always save (last epoch wins)
            torch.save({'model': model.state_dict(), 'in_ch': args.in_ch, 'base': args.base}, args.out)
            print(f"saved checkpoint to {args.out}")
        sched.step()

# -----------------------------
# Prediction helpers & visualizations
# -----------------------------

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 uint8 RGB for visualization from float [0,1] or uint8."""
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5,
                 outline: bool = False, thickness: int = 2) -> np.ndarray:
    """Overlay color where mask>0 on top of base image. color is RGB."""
    base = _to_uint8_rgb(base_rgb)
    mask01 = (mask > 0).astype(np.uint8)
    overlay = base.copy()
    # filled overlay
    col = np.array(color, dtype=np.uint8)[None, None, :]
    overlay_region = (mask01[..., None] * col) + ((1 - mask01[..., None]) * overlay)
    blended = (alpha * overlay_region + (1 - alpha) * overlay).astype(np.uint8)
    if outline:
        cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, tuple(int(c) for c in color), thickness=thickness)
    return blended


def apply_colormap(prob: np.ndarray) -> np.ndarray:
    """Convert prob [0,1] to an RGB heatmap (JET)."""
    p8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(p8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb


def absdiff_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel absolute difference in RGB, uint8."""
    a8 = _to_uint8_rgb(a)
    b8 = _to_uint8_rgb(b)
    # cv2.absdiff works on uint8 directly; keep as RGB array
    diff = cv2.absdiff(a8, b8)
    return diff


def absdiff_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Grayscale absolute difference (mean over channels), uint8."""
    af = normalize(a)
    bf = normalize(b)
    d = np.mean(np.abs(af - bf), axis=2)
    return np.clip(d * 255.0, 0, 255).astype(np.uint8)


def predict_prob(model: nn.Module, a: np.ndarray, b: np.ndarray, tile: int, overlap: int, device: str) -> np.ndarray:
    """Return probability map (float32, [0,1]) using sliding window if needed."""
    H, W, _ = a.shape
    if tile <= 0 or (H <= tile and W <= tile):
        ta = to_tensor(a).to(device)
        tb = to_tensor(b).to(device)
        with torch.no_grad():
            logits = model(ta, tb)
            prob = torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)
        return prob
    stride = tile - overlap
    out = np.zeros((H,W), dtype=np.float32)
    cnt = np.zeros((H,W), dtype=np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            yy = min(y+tile, H); xx = min(x+tile, W)
            y0 = yy - tile; x0 = xx - tile
            ca = a[y0:yy, x0:xx]; cb = b[y0:yy, x0:xx]
            ta = to_tensor(ca).to(device)
            tb = to_tensor(cb).to(device)
            with torch.no_grad():
                logits = model(ta, tb)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)
            out[y0:yy, x0:xx] += prob
            cnt[y0:yy, x0:xx] += 1.0
    prob = out / np.maximum(cnt, 1e-6)
    return prob



def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = normalize(img)
    return torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)


def predict_single(model: nn.Module, a: np.ndarray, b: np.ndarray, tile: int, overlap: int, thr: float, device: str) -> np.ndarray:
    H, W, _ = a.shape
    if tile <= 0 or (H <= tile and W <= tile):
        ta = to_tensor(a).to(device)
        tb = to_tensor(b).to(device)
        with torch.no_grad():
            logits = model(ta, tb)
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        mask = (prob >= thr).astype(np.uint8) * 255
        return mask
    stride = tile - overlap
    out = np.zeros((H,W), dtype=np.float32)
    cnt = np.zeros((H,W), dtype=np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            yy = min(y+tile, H); xx = min(x+tile, W)
            y0 = yy - tile; x0 = xx - tile
            ca = a[y0:yy, x0:xx]; cb = b[y0:yy, x0:xx]
            ta = to_tensor(ca).to(device)
            tb = to_tensor(cb).to(device)
            with torch.no_grad():
                logits = model(ta, tb)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
            out[y0:yy, x0:xx] += prob
            cnt[y0:yy, x0:xx] += 1.0
    prob = out / np.maximum(cnt, 1e-6)
    mask = (prob >= thr).astype(np.uint8) * 255
    return mask


def morphological_cleanup(mask: np.ndarray, min_blob: int = 0) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if min_blob > 0:
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), connectivity=8)
        sizes = stats[1:, -1]
        cleaned = np.zeros_like(mask)
        for i, s in enumerate(sizes, start=1):
            if s >= min_blob:
                cleaned[output == i] = 255
        mask = cleaned
    return mask


def predict_cmd(args):
    ckpt = torch.load(args.model, map_location='cpu')
    in_ch = ckpt.get('in_ch', args.in_ch)
    base  = ckpt.get('base', args.base)
    model = SiameseUNet(in_ch=in_ch, base=base)
    model.load_state_dict(ckpt['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device); model.eval()

    a = read_image(args.before, in_ch)
    b = read_image(args.after, in_ch)
    if a.shape != b.shape:
        raise SystemExit(f"Error: before {a.shape} and after {b.shape} must match. Use --align if needed.")

    a = normalize(a)
    b = normalize(b)
    if args.align:
        b = align_image(a, b)

    # probability map → mask
    prob = predict_prob(model, a, b, tile=args.tile, overlap=args.overlap, device=device)
    mask = (prob >= args.thr).astype(np.uint8) * 255
    mask = morphological_cleanup(mask, min_blob=args.min_blob)

    # save binary mask
    cv2.imwrite(args.out, mask)
    print(f"Saved change mask → {args.out}")

    # optional overlays / heatmap / diffs
    if args.overlay:
        base_img = a if args.overlay_on == 'before' else b
        overlay = make_overlay(base_img, mask, color=(255,0,0), alpha=args.overlay_alpha, outline=args.outline, thickness=args.thickness)
        cv2.imwrite(args.overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay → {args.overlay}")
    if args.heatmap:
        heat = apply_colormap(prob)
        cv2.imwrite(args.heatmap, cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
        print(f"Saved heatmap → {args.heatmap}")
    if args.diff:
        d_rgb = absdiff_rgb(a, b)
        cv2.imwrite(args.diff, cv2.cvtColor(d_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved RGB absdiff → {args.diff}")
    if args.diff_gray:
        d_gray = absdiff_gray(a, b)
        cv2.imwrite(args.diff_gray, d_gray)
        print(f"Saved gray absdiff → {args.diff_gray}")
    if args.diff_masked:
        d_rgb = absdiff_rgb(a, b)
        mask01 = (mask>0).astype(np.uint8)
        d_masked = (d_rgb * mask01[...,None]).astype(np.uint8)
        cv2.imwrite(args.diff_masked, cv2.cvtColor(d_masked, cv2.COLOR_RGB2BGR))
        print(f"Saved masked absdiff → {args.diff_masked}")

# -----------------------------
# Batch prediction
# -----------------------------

def predict_batch_cmd(args):
    """Predict masks for every row in a CSV and write them to an output folder.
    Optionally also write overlays/heatmaps/diffs to their own folders.
    """
    ckpt = torch.load(args.model, map_location='cpu')
    in_ch = ckpt.get('in_ch', args.in_ch)
    base  = ckpt.get('base', args.base)
    model = SiameseUNet(in_ch=in_ch, base=base)
    model.load_state_dict(ckpt['model'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device); model.eval()

    os.makedirs(args.outdir, exist_ok=True)
    overlay_dir = args.overlay_dir
    heatmap_dir = args.heatmap_dir
    diff_dir = args.diff_dir
    diff_gray_dir = args.diff_gray_dir
    diff_masked_dir = args.diff_masked_dir
    for d in [overlay_dir, heatmap_dir, diff_dir, diff_gray_dir, diff_masked_dir]:
        if d:
            os.makedirs(d, exist_ok=True)

    with open(args.csv, 'r') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows in {args.csv}")
        return

    for row in tqdm(rows, desc='predict-batch'):
        before = row['before']; after = row['after']
        try:
            a = read_image(before, in_ch); b = read_image(after, in_ch)
            if a.shape != b.shape:
                print(f"[skip] shape mismatch: {before} {a.shape} vs {after} {b.shape}")
                continue
            a = normalize(a); b = normalize(b)
            if args.align:
                b = align_image(a, b)
            prob = predict_prob(model, a, b, tile=args.tile, overlap=args.overlap, device=device)
            mask = (prob >= args.thr).astype(np.uint8) * 255
            mask = morphological_cleanup(mask, min_blob=args.min_blob)
            out_name = os.path.basename(before)
            cv2.imwrite(os.path.join(args.outdir, out_name), mask)

            if overlay_dir:
                base_img = a if args.overlay_on == 'before' else b
                overlay = make_overlay(base_img, mask, color=(255,0,0), alpha=args.overlay_alpha, outline=args.outline, thickness=args.thickness)
                cv2.imwrite(os.path.join(overlay_dir, out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if heatmap_dir:
                heat = apply_colormap(prob)
                cv2.imwrite(os.path.join(heatmap_dir, out_name), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
            if diff_dir or diff_gray_dir or diff_masked_dir:
                d_rgb = absdiff_rgb(a, b)
                if diff_dir:
                    cv2.imwrite(os.path.join(diff_dir, out_name), cv2.cvtColor(d_rgb, cv2.COLOR_RGB2BGR))
                if diff_gray_dir:
                    d_gray = absdiff_gray(a, b)
                    cv2.imwrite(os.path.join(diff_gray_dir, out_name), d_gray)
                if diff_masked_dir:
                    mask01 = (mask>0).astype(np.uint8)
                    d_masked = (d_rgb * mask01[...,None]).astype(np.uint8)
                    cv2.imwrite(os.path.join(diff_masked_dir, out_name), cv2.cvtColor(d_masked, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"[error] {before} / {after}: {e}")

# -----------------------------
# CSV utilities (LEVIR‑CD in CWD)
# -----------------------------

def _write_csv(rows: List[dict], out_csv: str):
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["before","after","mask"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


def make_csvs(root: str = "."):
    """Scan ROOT/{train,val,test}/A|B|label and write levir_*.csv in CWD.
    If not found directly under ROOT, tries one nested directory (e.g., ROOT/LEVIR-CD/train...).
    """
    def has_split(base: str) -> bool:
        return (
            os.path.isdir(os.path.join(base, "train", "A")) and
            os.path.isdir(os.path.join(base, "train", "B")) and
            os.path.isdir(os.path.join(base, "train", "label"))
        ) or (
            os.path.isdir(os.path.join(base, "val", "A")) and
            os.path.isdir(os.path.join(base, "val", "B")) and
            os.path.isdir(os.path.join(base, "val", "label"))
        ) or (
            os.path.isdir(os.path.join(base, "test", "A")) and
            os.path.isdir(os.path.join(base, "test", "B")) and
            os.path.isdir(os.path.join(base, "test", "label"))
        )

    root = os.path.abspath(root)
    scan_root = root
    if not has_split(scan_root):
        # try one-level nested
        for sub in sorted(os.listdir(root)):
            cand = os.path.join(root, sub)
            if os.path.isdir(cand) and has_split(cand):
                scan_root = cand
                print(f"Detected dataset under: {scan_root}")
                break

    exts = ["*.png", "*.PNG", "*.tif", "*.tiff", "*.jpg", "*.jpeg"]

    found_any = False
    for split in ["train", "val", "test"]:
        a_dir = os.path.join(scan_root, split, "A")
        b_dir = os.path.join(scan_root, split, "B")
        m_dir = os.path.join(scan_root, split, "label")
        if not (os.path.isdir(a_dir) and os.path.isdir(b_dir) and os.path.isdir(m_dir)):
            continue
        rows = []
        files = []
        for ext in exts:
            files.extend(sorted(glob.glob(os.path.join(a_dir, ext))))
        for a in files:
            base = os.path.basename(a)
            b = os.path.join(b_dir, base)
            m = os.path.join(m_dir, base)
            if os.path.exists(b) and os.path.exists(m):
                # Store paths relative to CWD to keep CSV portable
                rows.append({"before": os.path.relpath(a, start=os.getcwd()),
                             "after":  os.path.relpath(b, start=os.getcwd()),
                             "mask":   os.path.relpath(m, start=os.getcwd())})
        if rows:
            out_csv = f"levir_{split}.csv"
            _write_csv(rows, out_csv)
            found_any = True
        else:
            print(f"No rows found for split '{split}' under {scan_root}.")

    if not found_any:
        print("Nothing found. Ensure your CWD or --root has train/val/test with A, B, label subfolders.")

# -----------------------------
# Evaluation
# -----------------------------

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred>0).astype(np.uint8); gt = (gt>0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(max(union, 1))


def eval_cmd(args):
    # defaults: levir_val.csv, model.pt in CWD
    csv_path = args.csv
    ckpt = torch.load(args.model, map_location='cpu')
    model = SiameseUNet(in_ch=ckpt.get('in_ch',3), base=ckpt.get('base',32))
    model.load_state_dict(ckpt['model']); model.eval()

    ious = []
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            a = normalize(read_image(row['before'], ckpt.get('in_ch',3)))
            b = normalize(read_image(row['after'], ckpt.get('in_ch',3)))
            prob_mask = predict_single(model, a, b, tile=args.tile, overlap=args.overlap, thr=args.thr, device='cpu')
            pred = morphological_cleanup(prob_mask, min_blob=args.min_blob)
            gt = cv2.imread(row['mask'], cv2.IMREAD_GRAYSCALE)
            i = iou_score(pred, gt)
            ious.append(i)
            print(os.path.basename(row['before']), f"IoU={i:.3f}")
    print("Mean IoU:", float(np.mean(ious)) if ious else 0.0)

# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='Satellite Change Detection — Train / Predict / Evaluate (CWD)')
    sub = p.add_subparsers(dest='cmd', required=True)

    pcsv = sub.add_parser('make-csvs', help='Scan ROOT/{train,val,test}/A|B|label and write levir_*.csv in CWD')
    pcsv.add_argument('--root', default='.', help='Root folder containing train/val/test (default: CWD)')

    pt = sub.add_parser('train', help='Train on CSV (defaults to ./levir_train.csv)')
    pt.add_argument('--csv', default='levir_train.csv', help='CSV with columns: before,after,mask (default: levir_train.csv)')
    pt.add_argument('--in-ch', type=int, default=3)
    pt.add_argument('--tile', type=int, default=256)
    pt.add_argument('--batch-size', type=int, default=8)
    pt.add_argument('--epochs', type=int, default=20)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--alpha', type=float, default=0.7, help='BCE weight: loss = a*BCE + (1-a)*Dice')
    pt.add_argument('--base', type=int, default=32)
    pt.add_argument('--val-frac', type=float, default=0.2, help='0.0 uses the whole CSV for training with no internal val split')
    pt.add_argument('--out', default='model.pt')

    pp = sub.add_parser('predict', help='Predict a binary change mask for a before/after pair')
    pp.add_argument('--model', default='model.pt')
    pp.add_argument('--before', required=True)
    pp.add_argument('--after', required=True)
    pp.add_argument('--out', default='change_mask.png')
    pp.add_argument('--in-ch', type=int, default=3)
    pp.add_argument('--base', type=int, default=32)
    pp.add_argument('--tile', type=int, default=0, help='tile size for sliding-window prediction; 0 = whole image')
    pp.add_argument('--overlap', type=int, default=32, help='overlap pixels when tiling')
    pp.add_argument('--thr', type=float, default=0.5, help='probability threshold for binary mask')
    pp.add_argument('--align', action='store_true', help='try lightweight ECC alignment of AFTER to BEFORE')
    pp.add_argument('--min-blob', type=int, default=0, help='remove connected components smaller than this many pixels')
    # visualization
    pp.add_argument('--overlay', default='', help='Path to save overlay visualization (optional)')
    pp.add_argument('--overlay-on', choices=['before','after'], default='after', help='Which image to overlay on')
    pp.add_argument('--overlay-alpha', type=float, default=0.5, help='Overlay blend strength (0..1)')
    pp.add_argument('--outline', action='store_true', help='Draw contours on top of overlay')
    pp.add_argument('--thickness', type=int, default=2, help='Contour thickness if --outline is set')
    pp.add_argument('--heatmap', default='', help='Path to save probability heatmap (optional)')
    # differences
    pp.add_argument('--diff', default='', help='Path to save RGB absolute difference (optional)')
    pp.add_argument('--diff-gray', default='', help='Path to save grayscale absolute difference (optional)')
    pp.add_argument('--diff-masked', default='', help='Path to save RGB absdiff masked by predicted change (optional)')

    pb = sub.add_parser('predict-batch', help='Predict masks for all rows in a CSV into an output directory')
    pb.add_argument('--csv', default='levir_val.csv')
    pb.add_argument('--model', default='model.pt')
    pb.add_argument('--outdir', default='preds')
    pb.add_argument('--in-ch', type=int, default=3)
    pb.add_argument('--base', type=int, default=32)
    pb.add_argument('--tile', type=int, default=512)
    pb.add_argument('--overlap', type=int, default=32)
    pb.add_argument('--thr', type=float, default=0.5)
    pb.add_argument('--align', action='store_true')
    pb.add_argument('--min-blob', type=int, default=50)
    # visualization folders
    pb.add_argument('--overlay-dir', default='', help='Optional folder to save overlays (same filenames as masks)')
    pb.add_argument('--heatmap-dir', default='', help='Optional folder to save probability heatmaps')
    pb.add_argument('--overlay-on', choices=['before','after'], default='after')
    pb.add_argument('--overlay-alpha', type=float, default=0.5)
    pb.add_argument('--outline', action='store_true')
    pb.add_argument('--thickness', type=int, default=2)
    # diff folders
    pb.add_argument('--diff-dir', default='', help='Optional folder to save RGB absolute differences')
    pb.add_argument('--diff-gray-dir', default='', help='Optional folder to save grayscale absolute differences')
    pb.add_argument('--diff-masked-dir', default='', help='Optional folder to save masked RGB absolute differences')

    pe = sub.add_parser('eval', help='Evaluate IoU over a CSV (defaults to levir_val.csv)')
    pe.add_argument('--csv', default='levir_val.csv')
    pe.add_argument('--model', default='model.pt')
    pe.add_argument('--tile', type=int, default=512)
    pe.add_argument('--overlap', type=int, default=32)
    pe.add_argument('--thr', type=float, default=0.5)
    pe.add_argument('--min-blob', type=int, default=50)

    args = p.parse_args()
    if args.cmd == 'make-csvs':
        make_csvs(args.root)
    elif args.cmd == 'train':
        train_cmd(args)
    elif args.cmd == 'predict':
        predict_cmd(args)
    elif args.cmd == 'predict-batch':
        predict_batch_cmd(args)
    elif args.cmd == 'eval':
        eval_cmd(args)

if __name__ == '__main__':
    main()
