"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Thesis: Image Steganography via Diffusion Models
Stage : Dataset Preprocessing

For every image in the HuggingFace dataset
    imraj-rabbani/filtered-midfreq-imagenet

  1. Discard images whose shorter side < 150 px
  2. CenterCrop to square (shorter side) → Lanczos resize to 256 × 256
  3. Save as PNG  → ./processed_dataset/img_XXXXX.png
  4. Compute block-wise 8×8 DCT (offline, scipy — no grad needed here)
     → saved as ./processed_dataset/img_XXXXX_dct.npy
     → shape: (3, 32, 32, 8, 8)   [C, H//8, W//8, 8, 8]

Usage
─────
    pip install datasets pillow numpy scipy tqdm
    python preprocess.py
"""


import os
import math
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from PIL import Image
from scipy.fft import dctn
from datasets import load_dataset

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("./processed_dataset")
TARGET_SIZE   = 256          # final resolution (square)
MIN_SHORT_DIM = 150          # discard images below this threshold
DCT_BLOCK     = 8            # block size for DCT
NUM_BLOCKS    = TARGET_SIZE // DCT_BLOCK  # 32 blocks per spatial axis

HF_DATASET    = "imraj-rabbani/filtered-imagenet"
HF_SPLIT      = "train"      # change to "all" or None to get every split
# ──────────────────────────────────────────────────────────────────────────────


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Crop the image to a square using the shorter dimension as the side."""
    w, h = img.size
    side = min(w, h)
    left  = (w - side) // 2
    upper = (h - side) // 2
    return img.crop((left, upper, left + side, upper + side))


def compute_block_dct(img_array: np.ndarray) -> np.ndarray:
    """
    Compute block-wise 8×8 DCT-II for an RGB image.

    Parameters
    ----------
    img_array : np.ndarray  shape (H, W, 3), dtype float32, range [-1, 1]

    Returns
    -------
    dct_tensor : np.ndarray  shape (3, H//8, W//8, 8, 8), dtype float32
    """
    H, W, C = img_array.shape
    assert H % DCT_BLOCK == 0 and W % DCT_BLOCK == 0, \
        f"Image dims {H}×{W} not divisible by {DCT_BLOCK}"

    Bh = H // DCT_BLOCK   # 32
    Bw = W // DCT_BLOCK   # 32

    # (H, W, C) → (C, Bh, 8, Bw, 8) → (C, Bh, Bw, 8, 8)
    blocks = (img_array
              .transpose(2, 0, 1)               # (C, H, W)
              .reshape(C, Bh, DCT_BLOCK, Bw, DCT_BLOCK)
              .transpose(0, 1, 3, 2, 4))        # (C, Bh, Bw, 8, 8)

    # Apply 2-D DCT-II on the last two axes (the 8×8 block)
    dct_tensor = dctn(blocks, axes=(-2, -1), norm="ortho").astype(np.float32)
    return dct_tensor  # (3, 32, 32, 8, 8)


def estimate_disk_usage(directory: Path) -> str:
    """Return a human-readable string for total size of a directory."""
    total_bytes = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total_bytes < 1024:
            return f"{total_bytes:.1f} {unit}"
        total_bytes /= 1024
    return f"{total_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for stego diffusion fine-tuning")
    parser.add_argument("--output_dir", type=str,  default=str(OUTPUT_DIR))
    parser.add_argument("--min_dim",    type=int,   default=MIN_SHORT_DIM)
    parser.add_argument("--split",      type=str,   default=HF_SPLIT)
    parser.add_argument("--num_proc",   type=int,   default=4,
                        help="Workers for HuggingFace dataset loading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[preprocess] Loading dataset: {HF_DATASET} (split='{args.split}') …")
    dataset = load_dataset(HF_DATASET, split=args.split, verification_mode="no_checks")
    total_raw = len(dataset)
    print(f"[preprocess] Found {total_raw:,} images in the dataset.\n")

    # ── Resume logic ──────────────────────────────────────────────────────────
    existing = sorted(output_dir.glob("img_*.png"))
    n_saved = len(existing)
    n_discarded = 0
    print(f"[preprocess] Resuming from {n_saved} already saved images.")

    # Zero-pad index width to accommodate the full dataset size
    idx_width = max(5, math.floor(math.log10(total_raw)) + 1)

    for raw_idx, sample in enumerate(tqdm(dataset, total=total_raw, desc="Processing")):

        # HuggingFace image datasets expose images as PIL Images under "image"
        # (or sometimes "img" — handle both)
        pil_img = sample.get("image") or sample.get("img")
        if pil_img is None:
            # Fallback: grab the first value that is a PIL Image
            for v in sample.values():
                if isinstance(v, Image.Image):
                    pil_img = v
                    break

        if pil_img is None:
            n_discarded += 1
            continue

        # Ensure RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # ── Filter: shorter side < MIN_SHORT_DIM ─────────────────────────────
        w, h = pil_img.size
        if min(w, h) < args.min_dim:
            n_discarded += 1
            continue

        # ── CenterCrop → Lanczos resize to 256×256 ───────────────────────────
        pil_img = center_crop_to_square(pil_img)
        pil_img = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # ── Increment counter and build file paths ────────────────────────────
        n_saved += 1
        stem = f"img_{n_saved:0{idx_width}d}"
        png_path = output_dir / f"{stem}.png"
        npy_path = output_dir / f"{stem}_dct.npy"

        # ── Skip if already processed (resume logic) ──────────────────────────
        if png_path.exists() and npy_path.exists():
            continue

        # ── Save PNG ──────────────────────────────────────────────────────────
        pil_img.save(png_path, format="PNG", optimize=False)

        # ── Compute block-wise 8×8 DCT ────────────────────────────────────────
        # Normalise to [-1, 1] exactly as the DataLoader will do at training time
        img_np = np.array(pil_img, dtype=np.float32) / 127.5 - 1.0  # (256,256,3)
        dct_tensor = compute_block_dct(img_np)                        # (3,32,32,8,8)

        np.save(npy_path, dct_tensor)

    # ── Summary ───────────────────────────────────────────────────────────────
    disk = estimate_disk_usage(output_dir)

    print("\n" + "═" * 60)
    print("  PREPROCESSING COMPLETE")
    print("═" * 60)
    print(f"  Total images in dataset  : {total_raw:>10,}")
    print(f"  Discarded (too small/err): {n_discarded:>10,}")
    print(f"  Successfully saved       : {n_saved:>10,}")
    print(f"  Output directory         : {output_dir.resolve()}")
    print(f"  Estimated disk usage     : {disk}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()