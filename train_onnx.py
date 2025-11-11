import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import onnx
import numpy as np

# -----------------------------
# Dataset
# -----------------------------
class ChangeDetectionDataset(Dataset):
    def __init__(self, csv_file, patch_size=512):
        import csv as csv_lib
        self.rows = []
        with open(csv_file, 'r') as f:
            reader = csv_lib.DictReader(f)
            for row in reader:
                self.rows.append(row)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        before_path = row['before']
        after_path  = row['after']
        mask_path   = row['mask']

        before = cv2.imread(before_path)
        after  = cv2.imread(after_path)
        mask   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if before is None or after is None or mask is None:
            raise FileNotFoundError(f"Cannot read image: {before_path}, {after_path}, {mask_path}")

        # Random patch crop
        h, w, _ = before.shape
        ph, pw = self.patch_size, self.patch_size
        if h > ph and w > pw:
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)
            before = before[y:y+ph, x:x+pw]
            after  = after[y:y+ph, x:x+pw]
            mask   = mask[y:y+ph, x:x+pw]

        # Convert to tensor and normalize
        before = torch.tensor(before, dtype=torch.float32).permute(2,0,1)/255.0
        after  = torch.tensor(after, dtype=torch.float32).permute(2,0,1)/255.0
        mask   = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)/255.0

        return before, after, mask

# -----------------------------
# Hyperparameters / Paths
# -----------------------------
TRAIN_CSV = "levir_train.csv"
VAL_CSV   = "levir_val.csv"
BATCH_SIZE = 1          # small batch for MacBook
EPOCHS = 20
PATCH_SIZE = 512
LEARNING_RATE = 1e-3
ONNX_MODEL_PATH = "model.onnx"

# -----------------------------
# Lightweight UNet
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up2 = DoubleConv(128 + 64, 64)
        self.dconv_up1 = DoubleConv(64 + 32, 32)
        self.conv_last = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        x = self.upsample(conv3)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        x = self.conv_last(x)
        return x

# -----------------------------
# Dataset and DataLoader
# -----------------------------
train_dataset = ChangeDetectionDataset(TRAIN_CSV, patch_size=PATCH_SIZE)
val_dataset   = ChangeDetectionDataset(VAL_CSV, patch_size=PATCH_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train")
    for before, after, mask in pbar:
        before, after, mask = before.to(device), after.to(device), mask.to(device)
        x = torch.cat([before, after], dim=1)
        optimizer.zero_grad()
        output = model(x)
        if output.shape != mask.shape:
            mask = F.interpolate(mask, size=output.shape[2:], mode='nearest')
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for before, after, mask in val_loader:
            before, after, mask = before.to(device), after.to(device), mask.to(device)
            x = torch.cat([before, after], dim=1)
            output = model(x)
            if output.shape != mask.shape:
                mask = F.interpolate(mask, size=output.shape[2:], mode='nearest')
            val_loss += criterion(output, mask).item()
    print(f"Epoch {epoch+1} validation loss: {val_loss / len(val_loader):.4f}")

# -----------------------------
# Export to ONNX
# -----------------------------
model.eval()
dummy_input = torch.randn(1, 6, PATCH_SIZE, PATCH_SIZE, device=device)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    input_names=['input_before', 'input_after'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={'input_before': {0: 'batch'}, 'input_after': {0: 'batch'}, 'output': {0: 'batch'}}
)
print(f"Model exported to {ONNX_MODEL_PATH}")
