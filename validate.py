"""
validate.py
─────────────────────────────────────────────────────────────────────────────
Thesis: Image Steganography via Diffusion Models
Stage : Post-training validation

Answers two questions:
  1. Did fine-tuning shift the DCT distribution toward mid-band frequencies?
  2. Does the fine-tuned model produce better steganographic carriers?

What this script does
─────────────────────
  A. Generate N images from vanilla SD 1.5 (baseline)
  B. Generate N images from fine-tuned model (LoRA weights loaded)
  C. For each set:
       - Compute mid-band DCT energy ratio
       - Embed a fixed secret bitstream using QIM in DCT domain
       - Measure PSNR and SSIM between cover and stego
  D. Print a clear comparison table + PASS/FAIL verdict

Usage
─────
    python validate.py
    python validate.py --num_images 20 --output_dir ./validation_output
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from scipy.fft import dctn, idctn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID       = "runwayml/stable-diffusion-v1-5"
FINETUNED_DIR  = Path("./finetune_output/final")   # where unet.pt was saved
RESOLUTION     = 256
DCT_BLOCK      = 8
DDIM_STEPS     = 50
NUM_IMAGES     = 10     # images to generate per model (increase for more reliable stats)
OUTPUT_DIR     = Path("./validation_output")

# Mid-band definition (must match finetune.py)
MID_BAND_MIN = 3
MID_BAND_MAX = 10

# QIM embedding delta (quantization step for bit embedding)
QIM_DELTA = 5.0

# Bits to embed per image (keep small relative to capacity)
SECRET_BITS = 512
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  DCT utilities (scipy — offline, no grad needed here)
# ══════════════════════════════════════════════════════════════════════════════

def compute_block_dct(img_array: np.ndarray) -> np.ndarray:
    """
    Block-wise 8×8 DCT-II on a (H, W, 3) float32 image in [-1, 1].
    Returns shape (3, H//8, W//8, 8, 8).
    """
    H, W, C = img_array.shape
    Bh, Bw = H // DCT_BLOCK, W // DCT_BLOCK
    blocks = (img_array
              .transpose(2, 0, 1)
              .reshape(C, Bh, DCT_BLOCK, Bw, DCT_BLOCK)
              .transpose(0, 1, 3, 2, 4))
    return dctn(blocks, axes=(-2, -1), norm="ortho").astype(np.float32)


def compute_block_idct(dct_tensor: np.ndarray) -> np.ndarray:
    """
    Inverse of compute_block_dct.
    Input shape (3, Bh, Bw, 8, 8) → output (H, W, 3) float32.
    """
    C, Bh, Bw = dct_tensor.shape[:3]
    H, W = Bh * DCT_BLOCK, Bw * DCT_BLOCK
    pixels = idctn(dct_tensor, axes=(-2, -1), norm="ortho")
    return (pixels
            .transpose(0, 1, 3, 2, 4)
            .reshape(C, H, W)
            .transpose(1, 2, 0)
            .astype(np.float32))


def mid_band_energy_ratio(dct_tensor: np.ndarray) -> float:
    """
    Ratio of mid-band DCT energy to total energy.
    Higher = better carrier for steganography.
    dct_tensor shape: (3, Bh, Bw, 8, 8)
    """
    u = np.arange(DCT_BLOCK)
    v = np.arange(DCT_BLOCK)
    freq_sum = u[:, None] + v[None, :]   # (8, 8)

    mid_mask  = (freq_sum >= MID_BAND_MIN) & (freq_sum <= MID_BAND_MAX)
    total_energy = float(np.sum(dct_tensor ** 2))
    mid_energy   = float(np.sum(dct_tensor[:, :, :, mid_mask] ** 2))

    if total_energy < 1e-8:
        return 0.0
    return mid_energy / total_energy


# ══════════════════════════════════════════════════════════════════════════════
#  QIM steganography (embed + extract)
# ══════════════════════════════════════════════════════════════════════════════

def get_mid_band_positions(Bh: int, Bw: int) -> list:
    """
    Return list of (bh, bw, u, v) positions in the mid-band, all channels,
    sorted deterministically.
    """
    u = np.arange(DCT_BLOCK)
    v = np.arange(DCT_BLOCK)
    freq_sum = u[:, None] + v[None, :]
    mid_uvs = [(int(uu), int(vv))
               for uu in range(DCT_BLOCK)
               for vv in range(DCT_BLOCK)
               if MID_BAND_MIN <= freq_sum[uu, vv] <= MID_BAND_MAX]

    positions = []
    for c in range(3):
        for bh in range(Bh):
            for bw in range(Bw):
                for (uu, vv) in mid_uvs:
                    positions.append((c, bh, bw, uu, vv))
    return positions


def qim_embed(img_array: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """
    Embed bits into img_array using QIM on mid-band DCT coefficients.

    Strategy: use even/odd parity of floor(coeff / delta) to encode the bit.
      bit=0 → quantize coeff to nearest even multiple of delta
      bit=1 → quantize coeff to nearest odd multiple of delta

    This scheme works correctly for both positive and negative coefficients
    and is trivially reversible at extraction time.

    Parameters
    ----------
    img_array : (H, W, 3) float32 in [-1, 1]
    bits      : 1-D array of 0/1 integers, length <= capacity

    Returns
    -------
    stego_array : (H, W, 3) float32 in [-1, 1]
    """
    dct = compute_block_dct(img_array)           # (3, Bh, Bw, 8, 8)
    C, Bh, Bw = dct.shape[:3]
    positions = get_mid_band_positions(Bh, Bw)

    if len(bits) > len(positions):
        raise ValueError(f"Too many bits ({len(bits)}) for capacity ({len(positions)})")

    dct_stego = dct.copy()
    for i, bit in enumerate(bits):
        c, bh, bw, uu, vv = positions[i]
        coeff = float(dct_stego[c, bh, bw, uu, vv])

        # Find the nearest quantization index
        idx = int(np.floor(coeff / QIM_DELTA))

        # Current parity of the index
        current_parity = idx % 2  # 0 = even, 1 = odd (may be negative in Python)
        # Normalise to 0 or 1
        current_parity = abs(current_parity)

        if current_parity == bit:
            # Index already has correct parity — quantize normally
            quantized = idx * QIM_DELTA
        else:
            # Shift index by 1 toward the nearest correct-parity neighbour
            # Choose the direction that minimises distortion
            candidate_up   = (idx + 1) * QIM_DELTA
            candidate_down = (idx - 1) * QIM_DELTA
            if abs(candidate_up - coeff) <= abs(candidate_down - coeff):
                quantized = candidate_up
            else:
                quantized = candidate_down

        dct_stego[c, bh, bw, uu, vv] = quantized

    stego = compute_block_idct(dct_stego)
    return np.clip(stego, -1.0, 1.0)


def qim_extract(stego_array: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Extract bits from stego image.
    Reads the parity of floor(coeff / delta) for each embedding position.
    bit=0 if parity is even, bit=1 if parity is odd.
    """
    dct = compute_block_dct(stego_array)
    C, Bh, Bw = dct.shape[:3]
    positions = get_mid_band_positions(Bh, Bw)

    bits = []
    for i in range(num_bits):
        c, bh, bw, uu, vv = positions[i]
        coeff = float(dct[c, bh, bw, uu, vv])
        idx = int(np.floor(coeff / QIM_DELTA))
        bits.append(abs(idx % 2))   # 0 if even index, 1 if odd index
    return np.array(bits, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Image quality metrics
# ══════════════════════════════════════════════════════════════════════════════

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two images in [-1, 1]. Higher = less distortion."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    # Data range is 2.0 (from -1 to 1)
    return float(10.0 * math.log10((2.0 ** 2) / mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified SSIM (mean over channels). Range [0, 1], higher = more similar.
    img1, img2: (H, W, 3) float32 in [-1, 1]
    """
    C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    scores = []
    for c in range(3):
        x = img1[:, :, c].astype(np.float64)
        y = img2[:, :, c].astype(np.float64)
        mu_x, mu_y = x.mean(), y.mean()
        sig_x  = x.std()
        sig_y  = y.std()
        sig_xy = np.mean((x - mu_x) * (y - mu_y))
        num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2)
        scores.append(num / den)
    return float(np.mean(scores))


def bit_accuracy(original: np.ndarray, extracted: np.ndarray) -> float:
    """Fraction of bits correctly recovered."""
    return float(np.mean(original == extracted))


# ══════════════════════════════════════════════════════════════════════════════
#  Image generation
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(device, dtype, finetuned: bool, finetuned_dir: Path):
    """Load SD 1.5 pipeline, optionally with LoRA fine-tuned weights."""
    log.info(f"Loading {'fine-tuned' if finetuned else 'baseline'} model …")

    tokenizer   = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_enc    = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae         = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet        = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    scheduler   = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    text_enc = text_enc.to(dtype=dtype, device=device)
    vae      = vae.to(dtype=dtype, device=device)
    unet     = unet.to(dtype=dtype, device=device)

    if finetuned:
        # Apply LoRA config then load fine-tuned weights
        lora_config = LoraConfig(
            r              = 4,
            lora_alpha     = 8,
            target_modules = ["to_q", "to_k", "to_v"],
            lora_dropout   = 0.05,
            bias           = "none",
        )
        unet = get_peft_model(unet, lora_config)
        unet_weights = torch.load(finetuned_dir / "unet.pt", map_location=device)
        unet.load_state_dict(unet_weights, strict=False)
        log.info(f"Loaded fine-tuned UNet from {finetuned_dir / 'unet.pt'}")

        # Also load fine-tuned VAE decoder if present
        vae_path = finetuned_dir / "vae.pt"
        if vae_path.exists():
            vae.load_state_dict(torch.load(vae_path, map_location=device), strict=False)
            log.info(f"Loaded fine-tuned VAE from {vae_path}")

    text_enc.eval()
    vae.eval()
    unet.eval()

    return tokenizer, text_enc, vae, unet, scheduler


@torch.no_grad()
def generate_images(tokenizer, text_enc, vae, unet, scheduler,
                    device, dtype, num_images: int, seed: int = 42) -> list:
    """
    Generate num_images unconditional images using DDIM sampling.
    Returns list of (H, W, 3) float32 numpy arrays in [-1, 1].
    """
    # Null prompt embedding
    tokens = tokenizer(
        [""] * 1,
        padding        = "max_length",
        max_length     = tokenizer.model_max_length,
        truncation     = True,
        return_tensors = "pt",
    ).input_ids.to(device)
    text_emb = text_enc(tokens).last_hidden_state.to(dtype=dtype)  # (1, 77, 768)

    scheduler.set_timesteps(DDIM_STEPS)
    images = []
    generator = torch.Generator(device=device)

    for i in tqdm(range(num_images), desc="Generating"):
        generator.manual_seed(seed + i)

        # Start from random latent noise
        latents = torch.randn(
            1, 4, RESOLUTION // 8, RESOLUTION // 8,
            device=device, dtype=dtype, generator=generator
        )
        latents = latents * scheduler.init_noise_sigma

        # DDIM denoising loop
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t,
                              encoder_hidden_states=text_emb).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to pixels
        decoded = vae.decode(latents / vae.config.scaling_factor).sample  # (1,3,H,W)
        img = decoded[0].float().cpu().permute(1, 2, 0).numpy()           # (H,W,3)
        img = np.clip(img, -1.0, 1.0).astype(np.float32)
        images.append(img)

    return images


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_images(images: list, label: str, output_dir: Path) -> dict:
    """
    For a list of generated images:
      - Compute mid-band DCT energy ratio
      - Embed SECRET_BITS random bits using QIM
      - Measure PSNR, SSIM, bit accuracy
    Returns dict of mean metrics.
    """
    img_dir = output_dir / label
    img_dir.mkdir(parents=True, exist_ok=True)

    # Fixed secret bits — same for all images so comparison is fair
    rng = np.random.default_rng(0)
    secret_bits = rng.integers(0, 2, size=SECRET_BITS).astype(np.uint8)

    dct_ratios   = []
    psnr_vals    = []
    ssim_vals    = []
    bit_accs     = []
    capacities   = []

    for idx, img in enumerate(images):
        # Save cover image
        cover_pil = Image.fromarray(
            ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
        cover_pil.save(img_dir / f"cover_{idx:03d}.png")

        # DCT energy ratio
        dct = compute_block_dct(img)
        ratio = mid_band_energy_ratio(dct)
        dct_ratios.append(ratio)

        # Capacity (number of mid-band coefficients available)
        Bh, Bw = RESOLUTION // DCT_BLOCK, RESOLUTION // DCT_BLOCK
        positions = get_mid_band_positions(Bh, Bw)
        capacities.append(len(positions))

        # Embed secret bits
        try:
            stego = qim_embed(img, secret_bits)
        except ValueError as e:
            log.warning(f"Embedding skipped for image {idx}: {e}")
            continue

        # Save stego image
        stego_pil = Image.fromarray(
            ((stego + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
        stego_pil.save(img_dir / f"stego_{idx:03d}.png")

        # Quality metrics
        psnr_vals.append(psnr(img, stego))
        ssim_vals.append(ssim(img, stego))

        # Extraction accuracy
        extracted = qim_extract(stego, SECRET_BITS)
        bit_accs.append(bit_accuracy(secret_bits, extracted))

    return {
        "dct_mid_ratio"  : float(np.mean(dct_ratios)),
        "dct_mid_ratio_std": float(np.std(dct_ratios)),
        "capacity_bits"  : int(np.mean(capacities)),
        "psnr_db"        : float(np.mean(psnr_vals))   if psnr_vals  else 0.0,
        "ssim"           : float(np.mean(ssim_vals))   if ssim_vals  else 0.0,
        "bit_accuracy"   : float(np.mean(bit_accs))    if bit_accs   else 0.0,
        "n_images"       : len(images),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_images",    type=int,  default=NUM_IMAGES)
    p.add_argument("--output_dir",    type=str,  default=str(OUTPUT_DIR))
    p.add_argument("--finetuned_dir", type=str,  default=str(FINETUNED_DIR))
    p.add_argument("--seed",          type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    output_dir    = Path(args.output_dir)
    finetuned_dir = Path(args.finetuned_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not finetuned_dir.exists():
        log.error(f"Fine-tuned model not found at {finetuned_dir}")
        log.error("Run finetune.py first, then re-run this script.")
        sys.exit(1)

    SEP  = "═" * 65
    SEP2 = "─" * 65

    # ── Generate baseline images ───────────────────────────────────────────
    tok, enc, vae, unet, sched = load_pipeline(
        device, dtype, finetuned=False, finetuned_dir=finetuned_dir)
    baseline_images = generate_images(
        tok, enc, vae, unet, sched, device, dtype,
        num_images=args.num_images, seed=args.seed)
    del tok, enc, vae, unet, sched
    torch.cuda.empty_cache()

    # ── Generate fine-tuned images ─────────────────────────────────────────
    tok, enc, vae, unet, sched = load_pipeline(
        device, dtype, finetuned=True, finetuned_dir=finetuned_dir)
    finetuned_images = generate_images(
        tok, enc, vae, unet, sched, device, dtype,
        num_images=args.num_images, seed=args.seed)
    del tok, enc, vae, unet, sched
    torch.cuda.empty_cache()

    # ── Evaluate both sets ─────────────────────────────────────────────────
    log.info("Evaluating baseline images …")
    baseline_metrics = evaluate_images(baseline_images, "baseline", output_dir)

    log.info("Evaluating fine-tuned images …")
    finetuned_metrics = evaluate_images(finetuned_images, "finetuned", output_dir)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + SEP)
    print("  VALIDATION RESULTS")
    print(SEP)
    print(f"  Images evaluated : {args.num_images} per model")
    print(f"  Secret payload   : {SECRET_BITS} bits embedded per image")
    print(f"  Output saved to  : {output_dir.resolve()}")
    print(SEP2)

    # Table header
    print(f"  {'Metric':<35} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {SEP2}")

    metrics = [
        ("Mid-band DCT energy ratio",
         "dct_mid_ratio", "{:.4f}", True),
        ("PSNR cover→stego (dB)  [target >38]",
         "psnr_db", "{:.2f}", True),
        ("SSIM cover→stego       [target >0.95]",
         "ssim", "{:.4f}", True),
        ("Bit extraction accuracy [target >0.99]",
         "bit_accuracy", "{:.4f}", True),
        ("Embedding capacity (bits)",
         "capacity_bits", "{:,}", True),
    ]

    all_pass = True
    for label, key, fmt, higher_is_better in metrics:
        b = baseline_metrics[key]
        f = finetuned_metrics[key]
        delta = f - b
        delta_str = ("+" if delta >= 0 else "") + fmt.format(delta)
        improved = delta > 0 if higher_is_better else delta < 0
        marker = "✓" if improved else "✗"
        print(f"  {marker} {label:<33} "
              f"{fmt.format(b):>10} "
              f"{fmt.format(f):>12} "
              f"{delta_str:>10}")
        if not improved:
            all_pass = False

    print(SEP2)

    # DCT ratio std
    print(f"  {'Mid-band ratio std (baseline)':<35} "
          f"{baseline_metrics['dct_mid_ratio_std']:>10.4f}")
    print(f"  {'Mid-band ratio std (fine-tuned)':<35} "
          f"{finetuned_metrics['dct_mid_ratio_std']:>10.4f}")

    print(SEP)
    print("  VERDICT")
    print(SEP2)

    dct_improved = finetuned_metrics["dct_mid_ratio"] > baseline_metrics["dct_mid_ratio"]
    psnr_ok      = finetuned_metrics["psnr_db"] > 38.0
    ssim_ok      = finetuned_metrics["ssim"] > 0.95
    bit_acc_ok   = finetuned_metrics["bit_accuracy"] > 0.99

    if dct_improved:
        pct = (finetuned_metrics["dct_mid_ratio"] - baseline_metrics["dct_mid_ratio"]) \
              / baseline_metrics["dct_mid_ratio"] * 100
        print(f"  ✓ DCT mid-band energy increased by {pct:.1f}% — fine-tuning WORKED")
    else:
        print(f"  ✗ DCT mid-band energy did NOT increase — fine-tuning had no effect")

    if psnr_ok:
        print(f"  ✓ PSNR {finetuned_metrics['psnr_db']:.2f} dB > 38 dB threshold — "
              f"embedding distortion is acceptable")
    else:
        print(f"  ✗ PSNR {finetuned_metrics['psnr_db']:.2f} dB < 38 dB — "
              f"consider reducing QIM_DELTA or SECRET_BITS")

    if ssim_ok:
        print(f"  ✓ SSIM {finetuned_metrics['ssim']:.4f} > 0.95 — "
              f"stego image is visually indistinguishable from cover")
    else:
        print(f"  ✗ SSIM {finetuned_metrics['ssim']:.4f} < 0.95 — "
              f"visible distortion after embedding")

    if bit_acc_ok:
        print(f"  ✓ Bit accuracy {finetuned_metrics['bit_accuracy']:.4f} — "
              f"secret can be recovered reliably")
    else:
        print(f"  ✗ Bit accuracy {finetuned_metrics['bit_accuracy']:.4f} — "
              f"extraction is unreliable (check QIM_DELTA)")

    print(SEP)
    if dct_improved and psnr_ok and ssim_ok and bit_acc_ok:
        print("  OVERALL: ✓ PASS — thesis hypothesis supported")
        print("  Fine-tuning produced better steganographic carriers.")
    elif dct_improved:
        print("  OVERALL: ~ PARTIAL — DCT shift confirmed but embedding")
        print("  quality needs tuning. Adjust QIM_DELTA in validate.py.")
    else:
        print("  OVERALL: ✗ FAIL — consider more training steps or")
        print("  increasing lambda_dct (try --lambda_dct 0.5).")
    print(SEP + "\n")

    # Save raw metrics to file for thesis records
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("BASELINE\n")
        for k, v in baseline_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFINETUNED\n")
        for k, v in finetuned_metrics.items():
            f.write(f"  {k}: {v}\n")
    log.info(f"Raw metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()