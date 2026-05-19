"""
check.py
─────────────────────────────────────────────────────────────────────────────
Pre-flight sanity check before running finetune.py.
Verifies dataset integrity, tensor shapes, and estimates VRAM usage.

Usage
─────
    python check.py
"""

import os
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ─── Config (must match finetune.py) ─────────────────────────────────────────
DATASET_DIR  = Path("E:/processed_dataset")
RESOLUTION   = 256
DCT_SHAPE    = (3, 32, 32, 8, 8)
BATCH_SIZE   = 32
NUM_SPOT_CHECKS = 10   # how many random files to inspect
# ──────────────────────────────────────────────────────────────────────────────

SEP  = "─" * 60
PASS = "  ✓"
FAIL = "  ✗"
WARN = "  !"


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("  STEP 1 — Dataset directory")
print(SEP)

if not DATASET_DIR.exists():
    print(f"{FAIL} Directory not found: {DATASET_DIR.resolve()}")
    print("     → Run preprocess.py first, then re-run this script.")
    raise SystemExit(1)

png_files = sorted(DATASET_DIR.glob("img_*.png"))
npy_files = sorted(DATASET_DIR.glob("img_*_dct.npy"))

print(f"{PASS} Directory exists : {DATASET_DIR.resolve()}")
print(f"{PASS} PNG files found  : {len(png_files):,}")
print(f"{PASS} NPY files found  : {len(npy_files):,}")

if len(png_files) == 0:
    print(f"{FAIL} No PNG files found — preprocess.py may still be running or failed.")
    raise SystemExit(1)

if len(png_files) != len(npy_files):
    print(f"{WARN} Mismatch: {len(png_files)} PNGs vs {len(npy_files)} NPY files.")
    print("     → Some images may be missing their DCT cache.")
else:
    print(f"{PASS} PNG / NPY counts match")

# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print(f"  STEP 2 — Spot-check {NUM_SPOT_CHECKS} random samples")
print(SEP)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

sample_paths = random.sample(png_files, min(NUM_SPOT_CHECKS, len(png_files)))
errors = []

for png_path in sample_paths:
    npy_path = png_path.with_name(png_path.stem + "_dct.npy")
    tag = png_path.name

    # Check NPY exists
    if not npy_path.exists():
        errors.append(f"{tag}: missing .npy file")
        continue

    # Load image
    try:
        img = Image.open(png_path).convert("RGB")
        tensor = transform(img)
    except Exception as e:
        errors.append(f"{tag}: image load failed — {e}")
        continue

    # Check image shape
    if tensor.shape != (3, RESOLUTION, RESOLUTION):
        errors.append(f"{tag}: image shape {tuple(tensor.shape)} ≠ (3,{RESOLUTION},{RESOLUTION})")
        continue

    # Load DCT
    try:
        dct = np.load(npy_path).astype(np.float32)
    except Exception as e:
        errors.append(f"{tag}: npy load failed — {e}")
        continue

    # Check DCT shape
    if dct.shape != DCT_SHAPE:
        errors.append(f"{tag}: DCT shape {dct.shape} ≠ {DCT_SHAPE}")
        continue

    # Check for NaN / Inf in DCT
    if not np.isfinite(dct).all():
        errors.append(f"{tag}: DCT contains NaN or Inf values")
        continue

    print(f"{PASS} {tag}  image{tuple(tensor.shape)}  dct{dct.shape}  "
          f"range=[{tensor.min():.2f}, {tensor.max():.2f}]")

if errors:
    print()
    print(f"{FAIL} {len(errors)} error(s) found:")
    for e in errors:
        print(f"       {e}")
else:
    print(f"\n{PASS} All {len(sample_paths)} spot-checks passed")

# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("  STEP 3 — Disk usage")
print(SEP)

total_bytes = sum(f.stat().st_size for f in DATASET_DIR.rglob("*") if f.is_file())
print(f"{PASS} Total dataset size : {fmt_bytes(total_bytes)}")

avg_png = total_bytes / max(len(png_files) * 2, 1)  # approx per-pair
print(f"     Avg per image pair : {fmt_bytes(avg_png)}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("  STEP 4 — CUDA / GPU")
print(SEP)

if not torch.cuda.is_available():
    print(f"{FAIL} CUDA not available — training will be extremely slow on CPU.")
    raise SystemExit(1)

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory
free_vram  = total_vram - torch.cuda.memory_allocated(0)

print(f"{PASS} GPU              : {gpu_name}")
print(f"{PASS} Total VRAM       : {fmt_bytes(total_vram)}")
print(f"{PASS} Free VRAM now    : {fmt_bytes(free_vram)}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("  STEP 5 — VRAM estimate for finetune.py")
print(SEP)

# Rough estimates for SD 1.5 fine-tuning in bf16
# These are conservative approximations based on known model sizes
MB = 1024 ** 2

estimates = {
    "UNet (bf16, full weights in memory)" : 1700 * MB,
    "VAE (bf16, decoder grad-enabled)"    :  320 * MB,
    "CLIP text encoder (bf16, frozen)"    :  470 * MB,
    "LoRA adapters (rank=4, Q+K+V)"       :   20 * MB,
    "EMA shadow weights (LoRA params)"    :   20 * MB,
    "Activations + latents (batch=32)"    : 3500 * MB,
    "Optimizer states (AdamW, bf16 param)": 1200 * MB,
    "Gradients"                           : 1700 * MB,
    "DCT computation overhead"            :  400 * MB,
    "CUDA kernels + misc fragmentation"   :  500 * MB,
}

total_est = sum(estimates.values())

print("  Estimated VRAM breakdown (batch=32, bf16):")
print()
for label, val in estimates.items():
    bar_len = int(val / (100 * MB))
    bar = "█" * bar_len
    print(f"     {label:<45} {fmt_bytes(val):>9}  {bar}")

print()
print(f"     {'TOTAL ESTIMATE':<45} {fmt_bytes(total_est):>9}")
print(f"     {'AVAILABLE VRAM':<45} {fmt_bytes(total_vram):>9}")

margin = total_vram - total_est
print()
if margin >= 0:
    print(f"{PASS} Estimated headroom: {fmt_bytes(margin)}")
    if margin < 500 * MB:
        print(f"{WARN} Headroom is tight (<500MB). Consider batch_size=16 + grad_accum=2.")
    else:
        print(f"     Batch size 32 should be fine.")
else:
    print(f"{FAIL} Estimated VRAM needed ({fmt_bytes(total_est)}) EXCEEDS available "
          f"({fmt_bytes(total_vram)}) by {fmt_bytes(-margin)}")
    print()
    print("  ── Recommended fix ──────────────────────────────────────────")
    print("  In finetune.py, change:")
    print("    BATCH_SIZE         = 16   # was 32")
    print("    GRAD_ACCUM_STEPS   = 2    # add this constant")
    print()
    print("  Then in the training loop, accumulate gradients over 2 steps")
    print("  before calling optimizer.step(). Effective batch size stays 32.")
    print("  ─────────────────────────────────────────────────────────────")

# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("  STEP 6 — Quick DataLoader throughput test")
print(SEP)

import time
from torch.utils.data import Dataset, DataLoader

class _QuickDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = self.tf(Image.open(p).convert("RGB"))
        dct = torch.from_numpy(np.load(
            p.with_name(p.stem + "_dct.npy")).astype(np.float32))
        return img, dct

test_paths = png_files[:min(256, len(png_files))]
test_ds    = _QuickDataset(test_paths)
test_dl    = DataLoader(test_ds, batch_size=32, num_workers=4,
                        pin_memory=True, drop_last=False)

print(f"  Loading {len(test_paths)} samples with 4 workers …")
t0 = time.time()
batches_loaded = 0
for imgs, dcts in test_dl:
    batches_loaded += 1
elapsed = time.time() - t0

imgs_per_sec = len(test_paths) / elapsed
print(f"{PASS} Loaded {len(test_paths)} images in {elapsed:.1f}s  "
      f"({imgs_per_sec:.0f} img/s)")

steps_per_epoch = len(png_files) // BATCH_SIZE
epoch_time_min  = (steps_per_epoch / (imgs_per_sec / BATCH_SIZE)) / 60
print(f"     Full dataset ({len(png_files):,} images) would take "
      f"~{epoch_time_min:.1f} min/epoch at this throughput")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("═" * 60)
print("  PRE-FLIGHT SUMMARY")
print("═" * 60)

if errors:
    print(f"{FAIL} Dataset has {len(errors)} corrupt/missing file(s) — fix before training.")
elif margin < 0:
    print(f"{WARN} VRAM may be insufficient at batch=32. See recommendation above.")
    print(f"     Everything else looks good.")
else:
    print(f"{PASS} All checks passed. Ready to launch:")
    print()
    print("     python finetune.py --lambda_dct 0.1 --steps 80000 > training_log.txt 2>&1")

print("═" * 60 + "\n")