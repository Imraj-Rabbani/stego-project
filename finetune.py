"""
finetune.py
─────────────────────────────────────────────────────────────────────────────
Thesis: Image Steganography via Diffusion Models
Stage : Fine-tune Stable Diffusion v1.5 with DCT Auxiliary Loss

This script supports TWO loss modes (selected via --loss_mode):

──────────────────────────────────────────────────────────────────────────────
  Mode 1 (default): "mid_reward"  — directly maximize DSTG-eligible mid-band
                                    coefficients (matches K metric in validate.py)
──────────────────────────────────────────────────────────────────────────────
    L_total = L_diffusion − λ_mid · mean(score)

    where score is a smooth approximation of DSTG's hard eligibility test:
        Y(x̂₀)         : differentiable BT.601 luminance of x̂₀, level-shifted by 128
        coeffs        : block-wise 8×8 DCT of Y
        q             : coeffs / Q_TABLE        (continuous — no rounding)
        score[u,v]    : sigmoid((|q[u,v]| − 2) · sharpness)
        mean(score)   : averaged over batch · blocks · stable mid positions

    Stable mid positions = the 39 positions DSTG uses (non-DC, Q ≥ 8).
    Each score is in [0, 1] and saturates around |q| ≈ 4–5, so the model
    cannot win by inflating magnitudes endlessly — only by pushing more
    coefficients across the embedability threshold.

──────────────────────────────────────────────────────────────────────────────
  Mode 2 (legacy): "dct_match"  — original spectral matching loss
──────────────────────────────────────────────────────────────────────────────
    L_total = L_diffusion + λ_dct · L_DCT

    L_DCT       = Σ_{u,v} W(u,v) · (DCT(x̂₀)[u,v] − DCT(x₀)[u,v])²

    W(u,v) frequency mask
      DC  (u+v == 0)       : 0.1
      Mid (3 ≤ u+v ≤ 10)  : 1.0 … 2.0  (linearly interpolated)
      High (u+v > 10)      : 0.3
      Remaining low-freq   : 0.5  (1 ≤ u+v < 3)

In both modes:
    L_diffusion = MSE(ε, ε_θ(x_t, t))               ← standard DDPM loss
    x̂₀          = (x_t − √(1−ᾱ_t)·ε_θ) / √(ᾱ_t)   ← x₀ estimate (latent → VAE decode)

Model   : runwayml/stable-diffusion-v1-5  (pixel-space UNet only)
Adapter : LoRA  rank=4  on Q,K,V projections (via peft)
Precision: bf16
Optimizer: AdamW  lr=1e-5  weight_decay=1e-2
Scheduler: CosineAnnealingLR + 1 000-step linear warmup

Usage
─────
    pip install torch torchvision diffusers accelerate peft wandb tqdm pillow numpy
    # Default (new) mode — directly maximize DSTG-eligible mid coefficients:
    python finetune.py --loss_mode mid_reward --lambda_mid 0.01 --steps 80000
    # Legacy spectral matching mode:
    python finetune.py --loss_mode dct_match  --lambda_dct  0.1  --steps 80000
"""

import os
import sys
import math
import argparse
import logging
import ctypes
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Paths & defaults ─────────────────────────────────────────────────────────
DATASET_DIR = Path("E:/processed_dataset")
OUTPUT_DIR    = Path("./finetune_output")
MODEL_ID      = "runwayml/stable-diffusion-v1-5"

# ─── Training hyper-parameters (from thesis spec) ─────────────────────────────
RESOLUTION         = 256
BATCH_SIZE         = 2             # reduced from 32; effective batch = 4 x 8 = 32
GRAD_ACCUM_STEPS   = 4             # accumulate 8 micro-steps before each optimizer update
TRAIN_STEPS        = 80_000        # extend to 200 000 for full run
WARMUP_STEPS       = 1_000
LR                 = 1e-5
WEIGHT_DECAY       = 1e-2
GRAD_CLIP          = 1.0
EMA_DECAY          = 0.9999
CHECKPOINT_EVERY   = 2_000
LAMBDA_DCT_DEFAULT = 0.1           # weight for legacy "dct_match" loss
LAMBDA_MID_DEFAULT = 0.01          # weight for new "mid_reward" loss (much smaller!)
MID_SHARPNESS      = 2.0           # sigmoid sharpness for smooth eligibility
LORA_RANK          = 4
DATALOADER_WORKERS = 4
DCT_BLOCK          = 8
NUM_BLOCKS         = RESOLUTION // DCT_BLOCK   # 32

# ─── DSTG-aligned constants (must match validate.py / DCT_Adaptive.py) ────────
# The JPEG luminance quantisation table that DSTG uses for its |q|≥2 test.
# We use the SAME table here so the training loss directly tracks what
# validate.py measures as "K" (count of eligible mid-band coefficients).
_Q_TABLE_VALUES = [
    [ 3,  2,  2,  3,  4,  6,  8, 10],
    [ 2,  2,  3,  4,  5,  9, 10,  9],
    [ 3,  3,  4,  5,  6,  9, 11,  9],
    [ 3,  4,  5,  6,  8, 14, 13, 10],
    [ 4,  5,  7,  9, 11, 17, 16, 12],
    [ 5,  7,  9, 10, 13, 17, 18, 15],
    [10, 13, 12, 14, 16, 19, 19, 17],
    [14, 17, 18, 18, 19, 18, 19, 17],
]
ELIGIBILITY_THRESHOLD = 2.0        # matches DSTG's |q| ≥ 2 eligibility test
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Windows sleep prevention  (keeps PC awake for the full training run)
# ══════════════════════════════════════════════════════════════════════════════

def prevent_sleep():
    """
    Tell Windows not to sleep or turn off the display while training runs.
    Uses SetThreadExecutionState — no extra packages, no admin rights needed.
    Safe to call on non-Windows systems (silently does nothing).
    """
    if sys.platform == "win32":
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        log.info("[sleep] Sleep prevention enabled — PC will stay awake during training.")


def allow_sleep():
    """
    Restore normal Windows sleep behaviour after training finishes or crashes.
    Safe to call on non-Windows systems (silently does nothing).
    """
    if sys.platform == "win32":
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        log.info("[sleep] Sleep prevention released — normal power settings restored.")


# ══════════════════════════════════════════════════════════════════════════════
#  DCT utilities  (fully differentiable, PyTorch matrix-multiply based)
# ══════════════════════════════════════════════════════════════════════════════

def build_dct_basis(n: int = 8, device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
    """
    Build the orthonormal DCT-II basis matrix of size n×n.

    D[k, i] = c(k) * cos(π/n * (i+0.5) * k)
    where c(0) = 1/√n  and  c(k>0) = √(2/n)

    Returns  shape (n, n)
    """
    i = torch.arange(n, dtype=torch.float64)
    k = torch.arange(n, dtype=torch.float64)
    # Outer product: D[k, i]
    D = torch.cos(math.pi / n * (i[None, :] + 0.5) * k[:, None])
    # Orthonormal scale factors
    scale = torch.ones(n, dtype=torch.float64) * math.sqrt(2.0 / n)
    scale[0] = math.sqrt(1.0 / n)
    D = D * scale[:, None]
    return D.to(dtype=dtype, device=device)   # (8, 8)


def block_dct2d(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Differentiable block-wise 2-D DCT-II.

    Parameters
    ----------
    x     : (B, C, H, W)      pixel values in [-1, 1]
    basis : (8, 8)             precomputed DCT basis

    Returns
    -------
    coeff : (B, C, H//8, W//8, 8, 8)
    """
    B, C, H, W = x.shape
    Bh, Bw = H // DCT_BLOCK, W // DCT_BLOCK

    # Reshape into blocks: (B, C, Bh, 8, Bw, 8) → (B, C, Bh, Bw, 8, 8)
    blocks = (x
              .reshape(B, C, Bh, DCT_BLOCK, Bw, DCT_BLOCK)
              .permute(0, 1, 2, 4, 3, 5)
              .contiguous())                            # (B, C, Bh, Bw, 8, 8)

    # Apply DCT-II along rows then columns using einsum with the basis matrix
    # Row DCT:  coeff[..., k, j] = Σ_i basis[k,i] * block[..., i, j]
    after_rows = torch.einsum("ki,...ij->...kj", basis, blocks)   # (B,C,Bh,Bw,8,8)
    # Col DCT:  coeff[..., k, l] = Σ_j basis[l,j] * after_rows[...,k,j]
    coeff      = torch.einsum("lj,...kj->...kl", basis, after_rows)  # (B,C,Bh,Bw,8,8)

    return coeff


def build_freq_weight_mask(device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
    """
    Build the W(u,v) frequency weight mask of shape (8, 8).

    Rules (thesis spec):
      DC  (u+v == 0)             : 0.1
      low-freq (1 ≤ u+v < 3)    : 0.5
      mid-band (3 ≤ u+v ≤ 10)   : linearly interpolated 1.0 → 2.0
      high-freq (u+v > 10)       : 0.3
    """
    u = torch.arange(8, dtype=torch.float32)
    v = torch.arange(8, dtype=torch.float32)
    freq_sum = u[:, None] + v[None, :]          # (8, 8)

    W = torch.zeros(8, 8, dtype=torch.float32)

    dc_mask   = freq_sum == 0
    low_mask  = (freq_sum >= 1)  & (freq_sum < 3)
    mid_mask  = (freq_sum >= 3)  & (freq_sum <= 10)
    high_mask = freq_sum > 10

    W[dc_mask]   = 0.1
    W[low_mask]  = 0.5
    W[high_mask] = 0.3

    # Mid-band: linear interpolation from 1.0 (freq_sum=3) to 2.0 (freq_sum=10)
    mid_vals = 1.0 + (freq_sum[mid_mask] - 3.0) / 7.0   # maps [3,10] → [1,2]
    W[mid_mask] = mid_vals

    return W.to(dtype=dtype, device=device)      # (8, 8)


def build_q_table_tensor(device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
    """
    Build the JPEG luminance quantisation table (matches DSTG / validate.py).
    Shape (8, 8). This is the SAME table that validate.py's DSTG extractor uses
    for its |q| ≥ 2 eligibility test, so the training loss directly tracks
    the K metric reported during validation.
    """
    return torch.tensor(_Q_TABLE_VALUES, dtype=torch.float32).to(dtype=dtype,
                                                                  device=device)


def build_stable_mid_mask(q_table: torch.Tensor) -> torch.Tensor:
    """
    Build a (8, 8) boolean mask of DSTG's 39 "stable" mid-band positions:
    every non-DC position where Q[u,v] ≥ 8.  These are the only positions
    DSTG considers for embedding, so they are the only positions whose
    eligibility we reward.
    """
    mask = (q_table.float() >= 8.0)
    mask[0, 0] = False                        # exclude DC
    return mask.to(dtype=q_table.dtype, device=q_table.device)   # (8, 8), 0/1


# ══════════════════════════════════════════════════════════════════════════════
#  EMA helper
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()
                       if v.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.named_parameters():
            if v.requires_grad and k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into model (for inference / checkpointing)."""
        for k, v in model.named_parameters():
            if k in self.shadow:
                v.data.copy_(self.shadow[k])

    def restore_from(self, model: nn.Module, backup: dict):
        """Restore original (non-EMA) weights from a backup dict."""
        for k, v in model.named_parameters():
            if k in backup:
                v.data.copy_(backup[k])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state: dict):
        self.shadow = state


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class StegoDataset(Dataset):
    """
    Loads 256×256 PNG images and their precomputed _dct.npy tensors
    from ./processed_dataset/.

    __getitem__ returns
        image     : torch.Tensor  (3, 256, 256)  float32  range [-1, 1]
        dct_cache : torch.Tensor  (3, 32, 32, 8, 8)  float32
    """

    def __init__(self, root: Path):
        self.root  = root
        self.paths = sorted(root.glob("img_*.png"))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root}. Run preprocess.py first.")
        log.info(f"Dataset: {len(self.paths):,} images found in {root}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),                          # → [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),          # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        npy_path = img_path.with_name(img_path.stem + "_dct.npy")

        # ── Load image ────────────────────────────────────────────────────────
        pil_img = Image.open(img_path).convert("RGB")
        image   = self.transform(pil_img)               # (3, 256, 256) float32

        # ── Load cached DCT ───────────────────────────────────────────────────
        dct_cache = torch.from_numpy(
            np.load(npy_path).astype(np.float32)        # (3, 32, 32, 8, 8)
        )

        return image, dct_cache


# ══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_dct_loss(
    x_hat_0:      torch.Tensor,   # (B, 3, 256, 256)    denoised estimate (pixel space)
    coeff_target: torch.Tensor,   # (B, 3, 32, 32, 8, 8) precomputed DCT of clean x0
    dct_basis:    torch.Tensor,   # (8, 8)
    freq_mask:    torch.Tensor,   # (8, 8)
) -> torch.Tensor:
    """
    L_DCT = mean_over_batch [ S_{u,v} W(u,v) * (F(x_hat_0)[u,v] - F(x0)[u,v])^2 ]
    where F denotes block-wise 8x8 DCT.

    The target DCT coefficients F(x0) come from the precomputed .npy cache
    loaded by the DataLoader - no second VAE decode needed.
    Gradients flow only through coeff_pred (the x_hat_0 branch).
    """
    coeff_pred = block_dct2d(x_hat_0, dct_basis)   # (B, C, Bh, Bw, 8, 8)  <- has grad

    diff_sq    = (coeff_pred - coeff_target) ** 2   # (B, C, Bh, Bw, 8, 8)

    # Broadcast freq_mask (8, 8) over all leading dims
    weighted   = diff_sq * freq_mask                # (B, C, Bh, Bw, 8, 8)

    return weighted.mean()


def compute_mid_band_reward_loss(
    x_hat_0:    torch.Tensor,   # (B, 3, 256, 256)   denoised estimate in [-1, 1] RGB
    dct_basis:  torch.Tensor,   # (8, 8)
    q_table:    torch.Tensor,   # (8, 8)
    mid_mask:   torch.Tensor,   # (8, 8)            DSTG stable mid positions (0/1)
    sharpness:  float = MID_SHARPNESS,
) -> torch.Tensor:
    """
    Differentiable, DSTG-aligned mid-band eligibility reward.

    The validator (validate.py) measures K = count of mid-band positions per block
    where |round(coeff_Y / Q[u,v])| ≥ 2.  We replicate this test smoothly:

        Y(x̂₀)         : BT.601 luminance, scaled to [0, 255], level-shifted by 128
        coeff[u,v]    : 8×8 block DCT of Y                             (differentiable)
        q[u,v]        : coeff[u,v] / Q[u,v]                            (continuous)
        score[u,v]    : sigmoid((|q[u,v]| − 2) · sharpness)            (in [0, 1])
        reward        : mean of score over batch · blocks · mid_mask positions

    The returned loss is −reward, so MINIMIZING it MAXIMIZES the mean number of
    DSTG-eligible mid-band coefficients per block.

    Each per-position score saturates around |q| ≈ 4–5, so the model gains
    nothing by inflating coefficient magnitudes beyond what's needed to clear
    the embedability threshold.  This keeps the training signal stable and
    prevents the model from producing high-magnitude artefacts.
    """
    B = x_hat_0.shape[0]

    # ── BT.601 luminance, differentiable, in centered uint8-equivalent range ──
    # x_hat_0 is in [-1, 1] RGB; rescale to [0, 255] per channel, then take Y,
    # then level-shift by 128 to match DSTG's "pixel − 128" convention.
    R = x_hat_0[:, 0]
    G = x_hat_0[:, 1]
    Bc = x_hat_0[:, 2]
    R8 = (R + 1.0) * 127.5
    G8 = (G + 1.0) * 127.5
    B8 = (Bc + 1.0) * 127.5
    Y = 0.299 * R8 + 0.587 * G8 + 0.114 * B8       # (B, H, W)
    Y_centered = Y - 128.0                          # (B, H, W)

    # ── Block-DCT of Y (single-channel; same einsum recipe as block_dct2d) ────
    H, W = Y_centered.shape[-2:]
    Bh, Bw = H // DCT_BLOCK, W // DCT_BLOCK
    blocks = (Y_centered
              .reshape(B, Bh, DCT_BLOCK, Bw, DCT_BLOCK)
              .permute(0, 1, 3, 2, 4)
              .contiguous())                        # (B, Bh, Bw, 8, 8)
    after_rows = torch.einsum("ki,...ij->...kj", dct_basis, blocks)
    coeff      = torch.einsum("lj,...kj->...kl", dct_basis, after_rows)  # (B,Bh,Bw,8,8)

    # ── Continuous quantization index q = coeff / Q[u,v] ─────────────────────
    q_cont = coeff / q_table                        # (B, Bh, Bw, 8, 8)

    # ── Smooth eligibility score: sigmoid((|q| − 2) · sharpness) ─────────────
    score = torch.sigmoid((q_cont.abs() - ELIGIBILITY_THRESHOLD) * sharpness)
    score = score * mid_mask                        # zero out non-stable positions

    # ── Mean per-block eligible-count, averaged over batch ────────────────────
    # Sum across the (8, 8) positions gives a per-block count-like score,
    # then mean over batch and spatial blocks.
    per_block = score.sum(dim=(-2, -1))             # (B, Bh, Bw)
    reward    = per_block.mean()                    # scalar

    # Negate so that minimizing the loss maximizes the reward
    return -reward


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",  type=str,   default=str(DATASET_DIR))
    p.add_argument("--output_dir",   type=str,   default=str(OUTPUT_DIR))
    p.add_argument("--loss_mode",    type=str,   default="mid_reward",
                   choices=["mid_reward", "dct_match"],
                   help="'mid_reward' (NEW, default): maximize DSTG-eligible mid coeffs. "
                        "'dct_match' (legacy): spectral matching of training-set DCT.")
    p.add_argument("--lambda_mid",   type=float, default=LAMBDA_MID_DEFAULT,
                   help="Weight for mid_reward loss. Ablate: 0.005, 0.01, 0.05.")
    p.add_argument("--mid_sharpness", type=float, default=MID_SHARPNESS,
                   help="Sigmoid sharpness for smooth eligibility. 2.0 ≈ DSTG hard count.")
    p.add_argument("--lambda_dct",   type=float, default=LAMBDA_DCT_DEFAULT,
                   help="Weight of legacy dct_match loss. Ablate: 0.05, 0.1, 0.5, 1.0")
    p.add_argument("--steps",        type=int,   default=TRAIN_STEPS)
    p.add_argument("--lora_rank",    type=int,   default=LORA_RANK)
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint directory to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16          # native on RTX 4080 Super

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Prevent Windows from sleeping during training ──────────────────────
    prevent_sleep()

    try:
        _main_body(args, device, dtype, output_dir)
    finally:
        # Always restore normal sleep settings, even if training crashes
        allow_sleep()


def _main_body(args, device, dtype, output_dir):

    # ── Optional wandb logging ─────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project="stego-diffusion", config=vars(args))

    # ── Precompute DCT helpers (once, on GPU) ─────────────────────────────
    dct_basis = build_dct_basis(DCT_BLOCK, device=device, dtype=dtype)
    freq_mask = build_freq_weight_mask(device=device, dtype=dtype)
    q_table   = build_q_table_tensor(device=device, dtype=dtype)
    mid_mask  = build_stable_mid_mask(q_table)
    log.info(f"Loss mode: {args.loss_mode}  "
             f"(λ_mid={args.lambda_mid}, λ_dct={args.lambda_dct}, "
             f"sharpness={args.mid_sharpness})")

    # ── Load SD 1.5 components ─────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_ID}")
    tokenizer  = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_enc   = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae        = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet       = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Cast all components to bf16 and move to GPU
    text_enc = text_enc.to(dtype=dtype, device=device)
    vae      = vae.to(dtype=dtype, device=device)
    unet     = unet.to(dtype=dtype, device=device)

    # ── Freeze text encoder and VAE encoder ──────────────────────────────────────
    text_enc.requires_grad_(False)
    # The VAE encoder is frozen (we only need it for encoding x₀ to latents).
    # The VAE decoder must stay in the gradient graph so that the DCT loss
    # can propagate gradients back through: DCT loss → VAE decode → x̂₀ latent → UNet.
    # The second VAE decode (for the DCT target) has been removed — we use
    # the precomputed .npy cache instead, which eliminates the bottleneck.
    for name, param in vae.named_parameters():
        if "encoder" in name:
            param.requires_grad_(False)
        else:
            # Decoder stays in the computational graph
            param.requires_grad_(True)

    # ── Apply LoRA to UNet attention Q, K, V projections ──────────────────
    lora_config = LoraConfig(
        r             = args.lora_rank,
        lora_alpha    = args.lora_rank * 2,
        target_modules = ["to_q", "to_k", "to_v"],
        lora_dropout  = 0.05,
        bias          = "none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Unconditional text embedding (null prompt) ─────────────────────────
    null_tokens  = tokenizer(
        [""] * BATCH_SIZE,
        padding          = "max_length",
        max_length       = tokenizer.model_max_length,
        truncation       = True,
        return_tensors   = "pt",
    ).input_ids.to(device)

    with torch.no_grad():
        null_text_emb = text_enc(null_tokens).last_hidden_state  # (B, 77, 768)
    null_text_emb = null_text_emb.to(dtype=dtype)

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMA(unet, decay=EMA_DECAY)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # UNet LoRA params + VAE decoder params are trainable
    # (VAE decoder must stay in graph for DCT loss gradients to reach UNet)
    trainable_params = [p for p in unet.parameters() if p.requires_grad] + \
                       [p for p in vae.parameters()  if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    # ── LR Scheduler: cosine with linear warmup ────────────────────────────
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer          = optimizer,
        num_warmup_steps   = WARMUP_STEPS,
        num_training_steps = args.steps,
    )

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = StegoDataset(Path(args.dataset_dir))
    dataloader = DataLoader(
        dataset,
        batch_size        = BATCH_SIZE,
        shuffle           = True,
        num_workers       = DATALOADER_WORKERS,
        pin_memory        = False,
        persistent_workers = True,
        drop_last         = True,
    )

    # ── Optionally resume from checkpoint ─────────────────────────────────
    global_step = 0
    if args.resume:
        ckpt = Path(args.resume)
        log.info(f"Resuming from {ckpt}")
        unet.load_state_dict(torch.load(ckpt / "unet.pt",  map_location=device))
        vae.load_state_dict( torch.load(ckpt / "vae.pt",   map_location=device))
        optimizer.load_state_dict(torch.load(ckpt / "optim.pt", map_location=device))
        ema.load_state_dict(torch.load(ckpt / "ema.pt",    map_location=device))
        global_step = int((ckpt.name).split("step")[-1])

    # ── Training loop ──────────────────────────────────────────────────────
    log.info(f"Starting training: {args.steps} steps  λ_DCT={args.lambda_dct}")
    unet.train()
    vae.train()  # VAE decoder is in the gradient graph (encoder is frozen)

    data_iter    = iter(dataloader)
    progress_bar = tqdm(range(global_step, args.steps), desc="Training")

    for step in progress_bar:
        global_step = step + 1

        # ── Gradient accumulation: accumulate GRAD_ACCUM_STEPS micro-batches
        #    before each optimizer update. Effective batch size = 4 x 8 = 32.
        optimizer.zero_grad()
        accum_loss_diff = 0.0
        accum_loss_dct  = 0.0

        for accum_step in range(GRAD_ACCUM_STEPS):
            # ── Fetch micro-batch ─────────────────────────────────────────
            try:
                images, dct_cache = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, dct_cache = next(data_iter)

            images    = images.to(device=device, dtype=dtype)       # (B,3,256,256)
            dct_cache = dct_cache.to(device=device, dtype=dtype)    # (B,3,32,32,8,8)

            B = images.shape[0]

            # ── Encode to latent space (VAE encoder — frozen, no grad needed) ──
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor            # (B,4,32,32)

            # ── Sample noise and timestep ─────────────────────────────────
            noise  = torch.randn_like(latents)
            t      = torch.randint(0, noise_sched.config.num_train_timesteps,
                                   (B,), device=device, dtype=torch.long)

            # ── Forward diffusion: x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε ──────
            x_t = noise_sched.add_noise(latents, noise, t)          # (B,4,32,32)

            # ── UNet predicts noise  ε_θ(x_t, t) ─────────────────────────
            # Expand null_text_emb to match current batch size
            cond = null_text_emb[:B]
            noise_pred = unet(x_t, t, encoder_hidden_states=cond).sample  # (B,4,32,32)

            # ── Standard diffusion loss ───────────────────────────────────
            loss_diff = F.mse_loss(noise_pred.float(), noise.float())

            # ── Recover x̂₀ from noise prediction ─────────────────────────
            # x̂₀ = (x_t − √(1−ᾱ_t) · ε_θ) / √(ᾱ_t)
            alphas_cumprod = noise_sched.alphas_cumprod.to(device=device, dtype=dtype)
            sqrt_alpha_bar      = alphas_cumprod[t] ** 0.5               # (B,)
            sqrt_one_minus_abar = (1 - alphas_cumprod[t]) ** 0.5         # (B,)

            # Reshape scalars for broadcasting: (B,) → (B,1,1,1)
            sqrt_alpha_bar      = sqrt_alpha_bar[:, None, None, None]
            sqrt_one_minus_abar = sqrt_one_minus_abar[:, None, None, None]

            x_hat_0_latent = (x_t - sqrt_one_minus_abar * noise_pred) / sqrt_alpha_bar
            x_hat_0_latent = x_hat_0_latent / vae.config.scaling_factor  # (B,4,32,32)

            # ── Decode x̂₀ to pixel space — gradients MUST flow through here ──
            # (Do NOT wrap in torch.no_grad() — thesis requirement)
            x_hat_0_pixels = vae.decode(x_hat_0_latent).sample           # (B,3,256,256)

            # ── DCT auxiliary loss ────────────────────────────────────────
            # Two modes selectable via --loss_mode:
            #
            #   "mid_reward" (default, NEW):
            #       Directly maximize the count of DSTG-eligible mid-band
            #       coefficients in x̂₀.  Returns a negative scalar; minimizing
            #       it maximizes the smooth eligibility score.  No .npy cache
            #       needed — the reward is computed from x̂₀ alone.
            #
            #   "dct_match" (legacy):
            #       Squared error between x̂₀'s DCT and the precomputed clean-x₀
            #       DCT from the .npy cache, weighted by W(u,v).
            if args.loss_mode == "mid_reward":
                loss_aux = compute_mid_band_reward_loss(
                    x_hat_0_pixels, dct_basis, q_table, mid_mask,
                    sharpness=args.mid_sharpness,
                )
                lambda_aux = args.lambda_mid
            else:                                       # "dct_match"
                loss_aux = compute_dct_loss(
                    x_hat_0_pixels, dct_cache, dct_basis, freq_mask,
                )
                lambda_aux = args.lambda_dct

            # ── Total loss (scaled by accum steps for correct gradient magnitude) ──
            loss = (loss_diff + lambda_aux * loss_aux) / GRAD_ACCUM_STEPS
            loss.backward()

            accum_loss_diff += loss_diff.item()
            accum_loss_dct  += loss_aux.item()

        # ── Optimizer step (once per GRAD_ACCUM_STEPS micro-batches) ─────
        nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()
        lr_scheduler.step()

        # ── EMA update ────────────────────────────────────────────────────
        ema.update(unet)

        # Average losses over accumulation steps for logging
        loss_diff = torch.tensor(accum_loss_diff / GRAD_ACCUM_STEPS)
        loss_aux  = torch.tensor(accum_loss_dct  / GRAD_ACCUM_STEPS)
        lambda_aux = (args.lambda_mid if args.loss_mode == "mid_reward"
                      else args.lambda_dct)
        loss      = loss_diff + lambda_aux * loss_aux

        # ── Logging ───────────────────────────────────────────────────────
        # For "mid_reward" mode loss_aux is negative (reward); we ALSO log the
        # raw eligibility count = -loss_aux so the number is interpretable
        # (typical value: 5–20 eligible coefficients per block).
        eligible_per_block = (-loss_aux.item() if args.loss_mode == "mid_reward"
                              else None)
        log_dict = {
            "loss/total"    : loss.item(),
            "loss/diffusion": loss_diff.item(),
            "loss/aux"      : loss_aux.item(),
            "loss_mode"     : args.loss_mode,
            "lr"            : lr_scheduler.get_last_lr()[0],
            "step"          : global_step,
        }
        if eligible_per_block is not None:
            log_dict["mid_eligible_per_block"] = eligible_per_block

        postfix = {
            "loss":   f"{loss.item():.4f}",
            "l_diff": f"{loss_diff.item():.4f}",
        }
        if args.loss_mode == "mid_reward":
            postfix["elig"] = f"{eligible_per_block:.2f}"
        else:
            postfix["l_dct"] = f"{loss_aux.item():.4f}"
        progress_bar.set_postfix(postfix)

        if args.use_wandb:
            import wandb
            wandb.log(log_dict, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % CHECKPOINT_EVERY == 0:
            ckpt_dir = output_dir / f"checkpoint_step{global_step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save raw (non-EMA) weights
            torch.save(unet.state_dict(),       ckpt_dir / "unet.pt")
            torch.save(vae.state_dict(),        ckpt_dir / "vae.pt")
            torch.save(optimizer.state_dict(),  ckpt_dir / "optim.pt")
            torch.save(ema.state_dict(),        ckpt_dir / "ema.pt")

            # Also save EMA weights separately for inference
            ema_dir = ckpt_dir / "ema_weights"
            ema_dir.mkdir(exist_ok=True)
            torch.save(ema.shadow, ema_dir / "unet_ema.pt")

            log.info(f"[step {global_step}] Checkpoint saved → {ckpt_dir}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unet.state_dict(),       final_dir / "unet.pt")
    torch.save(vae.state_dict(),        final_dir / "vae.pt")
    torch.save(ema.state_dict(),        final_dir / "ema.pt")
    torch.save(ema.shadow,              final_dir / "unet_ema_weights.pt")
    log.info(f"Training complete. Final model saved to {final_dir}")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()