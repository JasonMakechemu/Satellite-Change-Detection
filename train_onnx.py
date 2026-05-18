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
    """
    PyTorch Dataset for binary change detection.

    Reads image-pair/mask triplets from a CSV file with columns:
      'before' — path to the pre-change RGB image
      'after'  — path to the post-change RGB image
      'mask'   — path to the binary ground-truth mask (white = changed)

    Each call to __getitem__ returns a randomly cropped patch of size
    (patch_size x patch_size) so that training sees varied spatial context
    without loading full images into GPU memory.
    """

    def __init__(self, csv_file: str, patch_size: int = 512):
        """
        Args:
            csv_file:   Path to a CSV file with 'before', 'after', 'mask' columns.
            patch_size: Height and width of the random crop returned per sample.
        """
        import csv as csv_lib
        self.rows = []
        with open(csv_file, 'r') as f:
            reader = csv_lib.DictReader(f)
            for row in reader:
                self.rows.append(row)
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        """
        Load one image triplet, apply a random spatial crop, and return
        normalised float tensors ready for the model.

        Returns:
            before: (3, H, W) float32 tensor in [0, 1]
            after:  (3, H, W) float32 tensor in [0, 1]
            mask:   (1, H, W) float32 tensor in [0, 1]  (1 = changed pixel)
        """
        row = self.rows[idx]
        before_path = row['before']
        after_path  = row['after']
        mask_path   = row['mask']

        # Load images with OpenCV (BGR colour order for RGB images, grayscale for mask)
        before = cv2.imread(before_path)
        after  = cv2.imread(after_path)
        mask   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if before is None or after is None or mask is None:
            raise FileNotFoundError(
                f"Cannot read image: {before_path}, {after_path}, {mask_path}"
            )

        # --- Random patch crop ---
        # Cropping to a fixed size keeps GPU memory usage predictable and acts
        # as a data-augmentation technique (different crops each epoch).
        h, w, _ = before.shape
        ph, pw  = self.patch_size, self.patch_size

        if h > ph and w > pw:
            # Sample a random top-left corner that keeps the crop within bounds
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)
            before = before[y:y+ph, x:x+pw]
            after  = after [y:y+ph, x:x+pw]
            mask   = mask  [y:y+ph, x:x+pw]

        # --- Convert to float32 tensors and normalise ---
        # permute(2,0,1) reorders (H,W,C) -> (C,H,W) as PyTorch expects
        before = torch.tensor(before, dtype=torch.float32).permute(2, 0, 1) / 255.0
        after  = torch.tensor(after,  dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Mask is (H,W); unsqueeze adds the channel dim to give (1,H,W)
        mask   = torch.tensor(mask,   dtype=torch.float32).unsqueeze(0) / 255.0

        return before, after, mask


# -----------------------------
# Hyperparameters / Paths
# -----------------------------

TRAIN_CSV        = "levir_train.csv"
VAL_CSV          = "levir_val.csv"
BATCH_SIZE       = 1           # kept small to run on a MacBook CPU/MPS
EPOCHS           = 20
PATCH_SIZE       = 512         # must match the tile size used at inference time
LEARNING_RATE    = 1e-3
ONNX_MODEL_PATH  = "model.onnx"


# -----------------------------
# Lightweight UNet
# -----------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive (Conv2d -> ReLU) blocks — the basic building block of UNet.

    Using two convolutions before each pooling step gives the network a larger
    effective receptive field without requiring deeper layers or larger kernels.
    padding=1 on 3x3 kernels preserves spatial dimensions (no shrinkage).
    inplace=True on ReLU reduces memory allocation overhead.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Lightweight 3-level UNet for binary change detection.

    Input:  6-channel tensor (before RGB concatenated with after RGB).
    Output: 1-channel logit map; apply sigmoid to obtain change probabilities.

    Architecture overview
    ---------------------
    Encoder (contracting path):
      conv1: 6  -> 32 channels  (full resolution)
      conv2: 32 -> 64 channels  (1/2 resolution after max-pool)
      conv3: 64 -> 128 channels (1/4 resolution after max-pool)  <- bottleneck

    Decoder (expanding path):
      up2: upsample + skip-connect conv2 -> 128+64 -> 64 channels
      up1: upsample + skip-connect conv1 ->  64+32 -> 32 channels
      final 1x1 conv:                        32    ->  1 channel (logit)

    Skip connections (torch.cat) pass fine-grained spatial detail from the
    encoder directly to the decoder, helping the network localise changes
    precisely rather than only using the coarse bottleneck features.

    The channel counts (32/64/128) are intentionally small for fast training
    on a CPU/MPS device while still learning meaningful representations.
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 1):
        """
        Args:
            in_ch:  Number of input channels (6 = 3 before + 3 after).
            out_ch: Number of output channels (1 for binary change mask).
        """
        super().__init__()

        # --- Encoder ---
        self.dconv_down1 = DoubleConv(in_ch, 32)    # full-res feature maps
        self.dconv_down2 = DoubleConv(32, 64)        # half-res feature maps
        self.dconv_down3 = DoubleConv(64, 128)       # quarter-res bottleneck

        # Shared pooling and upsampling operators (stateless, so one instance suffices)
        self.maxpool  = nn.MaxPool2d(2)
        # Bilinear upsampling is smoother than nearest-neighbour and avoids
        # the checkerboard artefacts that transposed convolutions can introduce
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Decoder ---
        # Input channels = upsampled bottleneck + skip connection from encoder
        self.dconv_up2 = DoubleConv(128 + 64, 64)   # bottleneck + conv2 skip
        self.dconv_up1 = DoubleConv(64  + 32, 32)   # up2 output  + conv1 skip

        # 1x1 convolution projects the final feature map to a single logit per pixel
        self.conv_last = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # --- Encoder: build feature maps at three scales ---
        conv1 = self.dconv_down1(x)                  # (B, 32, H,   W)
        conv2 = self.dconv_down2(self.maxpool(conv1)) # (B, 64, H/2, W/2)
        conv3 = self.dconv_down3(self.maxpool(conv2)) # (B,128, H/4, W/4)  bottleneck

        # --- Decoder: upsample and fuse with skip connections ---
        x = self.upsample(conv3)                      # (B,128, H/2, W/2)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))  # cat -> (B,192, H/2, W/2) -> (B,64, H/2, W/2)

        x = self.upsample(x)                          # (B, 64, H,   W)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))  # cat -> (B, 96, H,   W) -> (B,32, H,   W)

        # Final 1x1 conv produces raw logits (no activation; sigmoid applied by loss)
        x = self.conv_last(x)                         # (B,  1, H,   W)
        return x


# -----------------------------
# Dataset and DataLoader
# -----------------------------

train_dataset = ChangeDetectionDataset(TRAIN_CSV, patch_size=PATCH_SIZE)
val_dataset   = ChangeDetectionDataset(VAL_CSV,   patch_size=PATCH_SIZE)

# shuffle=True during training randomises sample order each epoch, reducing
# the risk of the model memorising patch sequences.
# num_workers=0 avoids multiprocessing issues on macOS (MPS/fork limitations).
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -----------------------------
# Model, Loss, Optimizer
# -----------------------------

# Select the best available compute device:
#   CUDA  — NVIDIA GPU (fastest)
#   MPS   — Apple Silicon GPU via Metal Performance Shaders
#   CPU   — fallback for any other hardware
device = torch.device(
    "cuda"  if torch.cuda.is_available()               else
    "mps"   if torch.backends.mps.is_available()       else
    "cpu"
)

model     = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# BCEWithLogitsLoss combines a sigmoid activation with binary cross-entropy in
# a numerically stable single operation. The model therefore outputs raw logits
# (no sigmoid), and this loss handles the activation internally.
criterion = nn.BCEWithLogitsLoss()


# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    # ---- Training phase ----
    model.train()  # enables dropout / batch-norm training behaviour
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train")

    for before, after, mask in pbar:
        # Move tensors to the target device
        before, after, mask = before.to(device), after.to(device), mask.to(device)

        # Concatenate before/after along the channel axis: (B,3,H,W)+(B,3,H,W) -> (B,6,H,W)
        x = torch.cat([before, after], dim=1)

        optimizer.zero_grad()   # clear gradients from the previous step
        output = model(x)       # forward pass -> (B,1,H,W) logits

        # If the model output is smaller than the mask (e.g. due to odd input sizes),
        # resize the mask to match rather than resizing the output, to preserve logit scale
        if output.shape != mask.shape:
            mask = F.interpolate(mask, size=output.shape[2:], mode='nearest')

        loss = criterion(output, mask)
        loss.backward()         # compute gradients
        optimizer.step()        # update weights

        pbar.set_postfix({"loss": loss.item()})

    # ---- Validation phase ----
    model.eval()   # disables dropout / switches batch-norm to running stats
    val_loss = 0.0

    with torch.no_grad():  # disable gradient tracking to save memory and time
        for before, after, mask in val_loader:
            before, after, mask = before.to(device), after.to(device), mask.to(device)
            x = torch.cat([before, after], dim=1)
            output = model(x)

            if output.shape != mask.shape:
                mask = F.interpolate(mask, size=output.shape[2:], mode='nearest')

            val_loss += criterion(output, mask).item()

    # Average loss over all validation batches
    print(f"Epoch {epoch+1} validation loss: {val_loss / len(val_loader):.4f}")


# -----------------------------
# Export to ONNX
# -----------------------------

model.eval()  # ensure batch-norm uses running stats, not batch stats, during export

# A dummy input is required by torch.onnx.export to trace the computation graph.
# Shape (1, 6, H, W): batch=1, channels=6 (before+after), spatial=PATCH_SIZE.
dummy_input = torch.randn(1, 6, PATCH_SIZE, PATCH_SIZE, device=device)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    # Descriptive I/O names make the exported graph easier to inspect in tools
    # like Netron and match the naming used at inference time
    input_names  = ['input_before', 'input_after'],
    output_names = ['output'],
    opset_version = 11,  # opset 11 is widely supported by ONNX Runtime versions
    # dynamic_axes allows the batch dimension to vary at inference time so the
    # same model file works for batch sizes other than 1 without re-exporting
    dynamic_axes = {
        'input_before': {0: 'batch'},
        'input_after':  {0: 'batch'},
        'output':        {0: 'batch'},
    },
)

print(f"Model exported to {ONNX_MODEL_PATH}")
