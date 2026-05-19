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
       - Compute mid-band DCT energy ratio (thesis hypothesis check)
       - Embed a fixed secret message using DSTG (adaptive DCT steganography)
       - Round-trip through PNG and extract the message
       - Measure PSNR, SSIM (skimage), and exact-recovery accuracy
  D. Print a clear comparison table + PASS/FAIL verdict

Embedding pipeline
──────────────────
  The embed/extract routines are the DSTG v1 pipeline (DCT_Adaptive.py +
  extractor.py), inlined here so validate.py stays self-contained:

    Color space : BGR → YCrCb; embed Y channel only.
    DCT         : 8×8 blocks, level-shift pixel−128, 2-D ortho DCT.
    Quantisation: JPEG luminance Q_TABLE, |q| ≥ 2 eligibility.
    Adaptive K  : mean (int) of eligible mid-freq counts + mean//2.
    Header      : 64 bits across 2 blocks, CRC14 protected.
    LSB match   : ±1 with min-distortion and never-below-2 guarantee.

Usage
─────
    python validate.py
    python validate.py --num_images 20 --output_dir ./validation_output
"""

import os
import sys
import io
import math
import zlib
import argparse
import logging
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

import cv2
from scipy.fft import dctn
from scipy.fftpack import dct as _scipy_dct, idct as _scipy_idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

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

# Bits to embed per image (must be a multiple of 8 — DSTG is byte-oriented)
SECRET_BITS = 512
assert SECRET_BITS % 8 == 0, "SECRET_BITS must be a multiple of 8 (DSTG embeds bytes)"
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  DSTG v1 — adaptive DCT steganography (inlined from DCT_Adaptive.py / extractor.py)
# ══════════════════════════════════════════════════════════════════════════════
#
# Embedder/extractor share the same constants, zigzag order, and Q_TABLE,
# so the extractor needs no key or sidecar file — it reads everything
# (payload length, adaptive K, version) from a 64-bit CRC-protected header
# embedded in blocks (0,0) and (0,1).

_DSTG_N                     = 8
_DSTG_VERSION               = 1
_DSTG_FALLBACK_K            = 24
_DSTG_ELIGIBILITY_THRESHOLD = 2
_DSTG_MAGIC_SHORT           = 0x4447
_DSTG_HEADER_BLOCKS         = [(0, 0), (0, 1)]
_DSTG_HEADER_BLOCK_BITS     = 32
_DSTG_HEADER_TOTAL_BITS     = 64

_DSTG_Q_TABLE = np.array([
    [ 3,  2,  2,  3,  4,  6,  8, 10],
    [ 2,  2,  3,  4,  5,  9, 10,  9],
    [ 3,  3,  4,  5,  6,  9, 11,  9],
    [ 3,  4,  5,  6,  8, 14, 13, 10],
    [ 4,  5,  7,  9, 11, 17, 16, 12],
    [ 5,  7,  9, 10, 13, 17, 18, 15],
    [10, 13, 12, 14, 16, 19, 19, 17],
    [14, 17, 18, 18, 19, 18, 19, 17],
], dtype=np.float32)


def _build_zigzag():
    order = []
    for s in range(2 * _DSTG_N - 1):
        if s % 2 == 0:
            r = min(s, _DSTG_N - 1); c = s - r
            while r >= 0 and c < _DSTG_N:
                order.append((r, c)); r -= 1; c += 1
        else:
            c = min(s, _DSTG_N - 1); r = s - c
            while c >= 0 and r < _DSTG_N:
                order.append((r, c)); r += 1; c -= 1
    return order


_DSTG_ZIGZAG = _build_zigzag()
_DSTG_ZZ_IDX = {uv: i for i, uv in enumerate(_DSTG_ZIGZAG)}

_DSTG_STABLE = [
    (u, v)
    for u in range(_DSTG_N) for v in range(_DSTG_N)
    if not (u == 0 and v == 0) and _DSTG_Q_TABLE[u, v] >= 8
]
_DSTG_PAYLOAD_CANDIDATES = sorted(_DSTG_STABLE, key=lambda uv: _DSTG_ZZ_IDX[uv])

_DSTG_HEADER_POSITIONS = sorted(
    _DSTG_STABLE,
    key=lambda uv: (-float(_DSTG_Q_TABLE[uv[0]][uv[1]]), uv[0], uv[1])
)[:_DSTG_HEADER_BLOCK_BITS]

_DSTG_HEADER_BLOCK_SET = set(_DSTG_HEADER_BLOCKS)


# ── 8×8 DCT helpers (separable; same as the scipy.fftpack reference) ─────────
def _dstg_dct2(block):
    return _scipy_dct(_scipy_dct(block.T, norm='ortho').T, norm='ortho')


def _dstg_idct2(block):
    return _scipy_idct(_scipy_idct(block.T, norm='ortho').T, norm='ortho')


def _dstg_to_blocks(channel):
    h, w   = channel.shape
    h8, w8 = h - h % _DSTG_N, w - w % _DSTG_N
    nr, nc = h8 // _DSTG_N, w8 // _DSTG_N
    ch     = channel[:h8, :w8].astype(np.float64)
    blocks = np.zeros((nr, nc, _DSTG_N, _DSTG_N), dtype=np.float64)
    for i in range(nr):
        for j in range(nc):
            blocks[i, j] = _dstg_dct2(
                ch[i * _DSTG_N:i * _DSTG_N + _DSTG_N,
                   j * _DSTG_N:j * _DSTG_N + _DSTG_N] - 128.0
            )
    return blocks, h8, w8


def _dstg_from_blocks(blocks, orig, h8, w8):
    out    = orig.copy()
    nr, nc = blocks.shape[:2]
    for i in range(nr):
        for j in range(nc):
            patch = np.clip(np.round(_dstg_idct2(blocks[i, j]) + 128.0), 0, 255)
            out[i * _DSTG_N:i * _DSTG_N + _DSTG_N,
                j * _DSTG_N:j * _DSTG_N + _DSTG_N] = patch.astype(np.uint8)
    return out


# ── Quantised LSB read / write ───────────────────────────────────────────────
def _dstg_qi(coeff, u, v):
    return int(np.round(float(coeff) / float(_DSTG_Q_TABLE[u, v])))


def _dstg_lsb_read(coeff, u, v):
    return _dstg_qi(coeff, u, v) & 1


def _dstg_lsb_write(coeff, u, v, bit, flip_dir=-1):
    Q = float(_DSTG_Q_TABLE[u, v])
    q = _dstg_qi(coeff, u, v)
    if (q & 1) == bit:
        return float(q) * Q
    q_up, q_dn = q + 1, q - 1
    up_ok = abs(q_up) >= _DSTG_ELIGIBILITY_THRESHOLD
    dn_ok = abs(q_dn) >= _DSTG_ELIGIBILITY_THRESHOLD
    if up_ok and dn_ok:
        if flip_dir == 1:
            return float(q_up) * Q
        if flip_dir == 0:
            return float(q_dn) * Q
        return (float(q_up) if abs(q_up * Q - coeff) <= abs(q_dn * Q - coeff)
                else float(q_dn)) * Q
    if up_ok:
        return float(q_up) * Q
    if dn_ok:
        return float(q_dn) * Q
    return float(q_up if q >= 0 else q_dn) * Q


def _dstg_make_prng(h, w):
    seed = (int(h) * 0x9e3779b9 ^ int(w) * 0x6c62272e) & 0xFFFFFFFF
    seed = seed or 0xDEADBEEF
    state = [seed]
    def _next():
        s = state[0]
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17)
        s ^= (s << 5)  & 0xFFFFFFFF
        state[0] = s
        return (s >> 16) & 1
    return _next


# ── Adaptive K and position selection ────────────────────────────────────────
def _dstg_payload_positions(K):
    return _DSTG_PAYLOAD_CANDIDATES[:K + 1]


def _dstg_compute_K(blocks):
    nr, nc = blocks.shape[:2]
    counts = [
        sum(1 for u, v in _DSTG_PAYLOAD_CANDIDATES
            if abs(_dstg_qi(blocks[i, j, u, v], u, v)) >= _DSTG_ELIGIBILITY_THRESHOLD)
        for i in range(nr) for j in range(nc)
        if (i, j) not in _DSTG_HEADER_BLOCK_SET
    ]
    if not counts:
        return _DSTG_FALLBACK_K
    mean_val = int(np.mean(counts))
    if mean_val <= 0:
        return _DSTG_FALLBACK_K
    extra = mean_val // 2
    return max(1, min(len(_DSTG_PAYLOAD_CANDIDATES) - 1, mean_val + extra))


# ── 64-bit header pack / unpack ──────────────────────────────────────────────
def _dstg_int_to_bits(value, n_bits):
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def _dstg_bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (int(b) & 1)
    return v


def _dstg_pack_header(n_message_bits, K):
    magic   = _DSTG_MAGIC_SHORT
    payload = n_message_bits & 0xFFFFFF
    k6      = K & 0x3F
    ver4    = _DSTG_VERSION & 0xF
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc14   = zlib.crc32(body) & 0x3FFF
    return (
        _dstg_int_to_bits(magic,   16)
        + _dstg_int_to_bits(payload, 24)
        + _dstg_int_to_bits(k6,       6)
        + _dstg_int_to_bits(ver4,     4)
        + _dstg_int_to_bits(crc14,   14)
    )


def _dstg_unpack_header(bits):
    magic   = _dstg_bits_to_int(bits[0:16])
    payload = _dstg_bits_to_int(bits[16:40])
    k6      = _dstg_bits_to_int(bits[40:46])
    ver4    = _dstg_bits_to_int(bits[46:50])
    crc14r  = _dstg_bits_to_int(bits[50:64])
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc_ok  = (zlib.crc32(body) & 0x3FFF) == crc14r
    magic_ok = magic == _DSTG_MAGIC_SHORT
    return payload, k6, ver4, (crc_ok and magic_ok)


def _dstg_write_header(blocks, n_message_bits, K):
    bits = _dstg_pack_header(n_message_bits, K)
    for blk_idx, (bi, bj) in enumerate(_DSTG_HEADER_BLOCKS):
        chunk = bits[blk_idx * _DSTG_HEADER_BLOCK_BITS:
                     (blk_idx + 1) * _DSTG_HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_DSTG_HEADER_POSITIONS):
            blocks[bi, bj, u, v] = _dstg_lsb_write(
                blocks[bi, bj, u, v], u, v, chunk[k], flip_dir=-1
            )


def _dstg_read_header(blocks):
    bits = []
    for bi, bj in _DSTG_HEADER_BLOCKS:
        for u, v in _DSTG_HEADER_POSITIONS:
            bits.append(_dstg_lsb_read(blocks[bi, bj, u, v], u, v))
    return _dstg_unpack_header(bits)


# ── DSTG public API ──────────────────────────────────────────────────────────
def dstg_get_capacity(cover_bgr):
    y            = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _dstg_to_blocks(y)
    K            = _dstg_compute_K(blocks)
    nr, nc       = blocks.shape[:2]
    n_payload    = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _DSTG_HEADER_BLOCK_SET
    )
    return n_payload * (K + 1)


def dstg_embed(cover_bgr, message):
    """
    Embed UTF-8 message into a uint8 BGR cover image.
    Returns (stego_bgr, info_dict).
    """
    h, w = cover_bgr.shape[:2]

    raw  = message.encode('utf-8')
    bits = [(b >> (7 - k)) & 1 for b in raw for k in range(8)]
    n    = len(bits)

    ycrcb     = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    blocks, h8, w8 = _dstg_to_blocks(y)
    K         = _dstg_compute_K(blocks)
    nr, nc    = blocks.shape[:2]

    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _DSTG_HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)

    if n > capacity:
        raise ValueError(
            f"Message too large: {n} bits needed, {capacity} available."
        )

    _dstg_write_header(blocks, n, K)

    prng    = _dstg_make_prng(h, w)
    bit_idx = 0
    for i in range(nr):
        for j in range(nc):
            if (i, j) in _DSTG_HEADER_BLOCK_SET:
                continue
            for u, v in _dstg_payload_positions(K):
                if bit_idx >= n:
                    break
                blocks[i, j, u, v] = _dstg_lsb_write(
                    blocks[i, j, u, v], u, v, bits[bit_idx], prng()
                )
                bit_idx += 1
            if bit_idx >= n:
                break

    stego_y   = _dstg_from_blocks(blocks, y, h8, w8)
    stego_bgr = cv2.cvtColor(cv2.merge([stego_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    info = {"K": K, "capacity_bits": capacity, "payload_bits": n}
    return stego_bgr, info


def dstg_extract(stego_bgr):
    """Extract UTF-8 message from a uint8 BGR stego image. Raises on failure."""
    ycrcb          = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb)
    y              = cv2.split(ycrcb)[0]
    blocks, h8, w8 = _dstg_to_blocks(y)

    n_bits, K, version, header_ok = _dstg_read_header(blocks)
    if not header_ok:
        raise ValueError("Header CRC/magic failed — no DSTG payload found.")

    nr, nc    = blocks.shape[:2]
    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _DSTG_HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)
    if not (0 < n_bits <= capacity):
        raise ValueError(f"Header payload_bits={n_bits} out of range [1, {capacity}].")

    bits = []
    for i in range(nr):
        for j in range(nc):
            if (i, j) in _DSTG_HEADER_BLOCK_SET:
                continue
            for u, v in _dstg_payload_positions(K):
                if len(bits) >= n_bits:
                    break
                bits.append(_dstg_lsb_read(blocks[i, j, u, v], u, v))
            if len(bits) >= n_bits:
                break

    byte_arr = bytearray(
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, len(bits) - 7, 8)
    )
    try:
        return byte_arr.decode('utf-8'), bits
    except UnicodeDecodeError as exc:
        raise ValueError("UTF-8 decode failed — bit-stream desync.") from exc


# ══════════════════════════════════════════════════════════════════════════════
#  DCT utilities (for the mid-band energy hypothesis check — separate from DSTG)
# ══════════════════════════════════════════════════════════════════════════════

def compute_block_dct(img_array: np.ndarray) -> np.ndarray:
    """
    Block-wise 8×8 DCT-II on a (H, W, 3) float32 image in [-1, 1].
    Used only for analysing the cover's frequency distribution.
    Returns shape (3, H//8, W//8, 8, 8).
    """
    H, W, C = img_array.shape
    Bh, Bw = H // DCT_BLOCK, W // DCT_BLOCK
    blocks = (img_array
              .transpose(2, 0, 1)
              .reshape(C, Bh, DCT_BLOCK, Bw, DCT_BLOCK)
              .transpose(0, 1, 3, 2, 4))
    return dctn(blocks, axes=(-2, -1), norm="ortho").astype(np.float32)


def mid_band_energy_ratio(dct_tensor: np.ndarray) -> float:
    """
    Ratio of mid-band DCT energy to total energy.
    Higher = better carrier for steganography.
    dct_tensor shape: (3, Bh, Bw, 8, 8)
    """
    u = np.arange(DCT_BLOCK)
    v = np.arange(DCT_BLOCK)
    freq_sum = u[:, None] + v[None, :]   # (8, 8)

    mid_mask     = (freq_sum >= MID_BAND_MIN) & (freq_sum <= MID_BAND_MAX)
    total_energy = float(np.sum(dct_tensor ** 2))
    mid_energy   = float(np.sum(dct_tensor[:, :, :, mid_mask] ** 2))

    if total_energy < 1e-8:
        return 0.0
    return mid_energy / total_energy


# ══════════════════════════════════════════════════════════════════════════════
#  Image format conversions
# ══════════════════════════════════════════════════════════════════════════════

def float_rgb_to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 [-1,1] RGB  →  (H,W,3) uint8 BGR (OpenCV convention)."""
    rgb = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def uint8_bgr_to_float_rgb(img: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 BGR  →  (H,W,3) float32 [-1,1] RGB."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 127.5) - 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Quality metrics (skimage, on uint8 BGR — matches DSTG.quality_metrics)
# ══════════════════════════════════════════════════════════════════════════════

def psnr_bgr(cover_bgr: np.ndarray, stego_bgr: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(cover_bgr, stego_bgr, data_range=255))


def ssim_bgr(cover_bgr: np.ndarray, stego_bgr: np.ndarray) -> float:
    return float(structural_similarity(
        cover_bgr, stego_bgr, data_range=255, channel_axis=2
    ))


def bit_accuracy(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Fraction of bits correctly recovered (handles length mismatch)."""
    n = min(len(original_bits), len(extracted_bits))
    if n == 0:
        return 0.0
    return float(np.mean(original_bits[:n] == extracted_bits[:n]))


# ══════════════════════════════════════════════════════════════════════════════
#  Secret message generation (DSTG is byte-oriented)
# ══════════════════════════════════════════════════════════════════════════════

def make_secret(n_bits: int, seed: int = 0) -> tuple:
    """
    Build a deterministic UTF-8 secret of exactly n_bits bits (= n_bits/8 bytes).
    Uses printable ASCII only so the message is a valid UTF-8 string.
    Returns (message_str, bits_array).
    """
    assert n_bits % 8 == 0, "n_bits must be a multiple of 8"
    n_bytes = n_bits // 8
    rng = np.random.default_rng(seed)
    # Printable ASCII range 33..126 (avoids whitespace ambiguities)
    chars = rng.integers(33, 127, size=n_bytes, dtype=np.int32)
    message = ''.join(chr(int(c)) for c in chars)
    raw  = message.encode('utf-8')
    bits = np.array(
        [(b >> (7 - k)) & 1 for b in raw for k in range(8)],
        dtype=np.uint8
    )
    return message, bits


# ══════════════════════════════════════════════════════════════════════════════
#  Image generation
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(device, dtype, finetuned: bool, finetuned_dir: Path):
    """Load SD 1.5 pipeline, optionally with LoRA fine-tuned weights."""
    log.info(f"Loading {'fine-tuned' if finetuned else 'baseline'} model …")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_enc  = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae       = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet      = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    text_enc = text_enc.to(dtype=dtype, device=device)
    vae      = vae.to(dtype=dtype, device=device)
    unet     = unet.to(dtype=dtype, device=device)

    if finetuned:
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
    Returns list of (H, W, 3) float32 numpy arrays in [-1, 1], RGB order.
    """
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

        latents = torch.randn(
            1, 4, RESOLUTION // 8, RESOLUTION // 8,
            device=device, dtype=dtype, generator=generator
        )
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            noise_pred = unet(latents, t,
                              encoder_hidden_states=text_emb).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        decoded = vae.decode(latents / vae.config.scaling_factor).sample  # (1,3,H,W)
        img = decoded[0].float().cpu().permute(1, 2, 0).numpy()           # (H,W,3) RGB
        img = np.clip(img, -1.0, 1.0).astype(np.float32)
        images.append(img)

    return images


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation pipeline (now using DSTG)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_images(images: list, label: str, output_dir: Path) -> dict:
    """
    For a list of generated images:
      - Compute mid-band DCT energy ratio (on the float32 RGB array)
      - Embed a fixed secret UTF-8 message using DSTG
      - Save cover & stego as PNG, then RE-LOAD the stego PNG and extract
        (this validates true round-trip robustness, not just in-memory)
      - Record PSNR, SSIM, bit accuracy, exact-recovery
    Returns dict of mean metrics.
    """
    img_dir = output_dir / label
    img_dir.mkdir(parents=True, exist_ok=True)

    # Fixed secret — same content for all images so comparison is fair
    secret_msg, secret_bits = make_secret(SECRET_BITS, seed=0)

    dct_ratios    = []
    psnr_vals     = []
    ssim_vals     = []
    bit_accs      = []
    exact_recover = []
    capacities    = []
    K_values      = []

    for idx, img in enumerate(tqdm(images, desc=f"Eval {label}")):
        # ── Convert float-RGB ↔ uint8-BGR for DSTG ──────────────────────────
        cover_bgr = float_rgb_to_uint8_bgr(img)

        # Save cover
        cover_path = img_dir / f"cover_{idx:03d}.png"
        cv2.imwrite(str(cover_path), cover_bgr)

        # ── Mid-band DCT energy ratio (thesis hypothesis check) ──────────────
        dct   = compute_block_dct(img)
        ratio = mid_band_energy_ratio(dct)
        dct_ratios.append(ratio)

        # ── Capacity (DSTG adaptive) ────────────────────────────────────────
        try:
            cap = dstg_get_capacity(cover_bgr)
            capacities.append(cap)
        except Exception as e:
            log.warning(f"[{label} {idx}] capacity check failed: {e}")
            capacities.append(0)
            continue

        # ── Embed ────────────────────────────────────────────────────────────
        try:
            stego_bgr, info = dstg_embed(cover_bgr, secret_msg)
            K_values.append(info["K"])
        except Exception as e:
            log.warning(f"[{label} {idx}] embedding skipped: {e}")
            continue

        # Save stego as PNG (lossless round-trip)
        stego_path = img_dir / f"stego_{idx:03d}.png"
        cv2.imwrite(str(stego_path), stego_bgr)

        # ── Reload stego from disk and extract — true round-trip test ──────
        stego_reloaded = cv2.imread(str(stego_path))
        try:
            recovered_msg, recovered_bits_list = dstg_extract(stego_reloaded)
            recovered_bits = np.array(recovered_bits_list, dtype=np.uint8)
            acc            = bit_accuracy(secret_bits, recovered_bits)
            exact          = 1.0 if recovered_msg == secret_msg else 0.0
        except Exception as e:
            log.warning(f"[{label} {idx}] extraction failed: {e}")
            acc   = 0.0
            exact = 0.0

        bit_accs.append(acc)
        exact_recover.append(exact)

        # ── Quality metrics on uint8 BGR (matches DSTG convention) ──────────
        psnr_vals.append(psnr_bgr(cover_bgr, stego_bgr))
        ssim_vals.append(ssim_bgr(cover_bgr, stego_bgr))

    return {
        "dct_mid_ratio"     : float(np.mean(dct_ratios))          if dct_ratios    else 0.0,
        "dct_mid_ratio_std" : float(np.std(dct_ratios))           if dct_ratios    else 0.0,
        "capacity_bits"     : int(np.mean(capacities))            if capacities    else 0,
        "K_mean"            : float(np.mean(K_values))            if K_values      else 0.0,
        "psnr_db"           : float(np.mean(psnr_vals))           if psnr_vals     else 0.0,
        "ssim"              : float(np.mean(ssim_vals))           if ssim_vals     else 0.0,
        "bit_accuracy"      : float(np.mean(bit_accs))            if bit_accs      else 0.0,
        "exact_recovery"    : float(np.mean(exact_recover))       if exact_recover else 0.0,
        "n_images"          : len(images),
        "n_embedded"        : len(bit_accs),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_images",    type=int, default=NUM_IMAGES)
    p.add_argument("--output_dir",    type=str, default=str(OUTPUT_DIR))
    p.add_argument("--finetuned_dir", type=str, default=str(FINETUNED_DIR))
    p.add_argument("--seed",          type=int, default=42)
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
    print("  VALIDATION RESULTS  (DSTG adaptive DCT pipeline)")
    print(SEP)
    print(f"  Images evaluated : {args.num_images} per model")
    print(f"  Secret payload   : {SECRET_BITS} bits ({SECRET_BITS // 8} chars) per image")
    print(f"  Round-trip       : embed → PNG → reload → extract")
    print(f"  Output saved to  : {output_dir.resolve()}")
    print(SEP2)

    print(f"  {'Metric':<38} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {SEP2}")

    metrics = [
        ("Mid-band DCT energy ratio",
         "dct_mid_ratio",  "{:.4f}", True),
        ("Adaptive K (mean)",
         "K_mean",         "{:.1f}", True),
        ("PSNR cover→stego (dB)  [target >38]",
         "psnr_db",        "{:.2f}", True),
        ("SSIM cover→stego       [target >0.95]",
         "ssim",           "{:.4f}", True),
        ("Bit accuracy            [target >0.99]",
         "bit_accuracy",   "{:.4f}", True),
        ("Exact recovery rate     [target =1.00]",
         "exact_recovery", "{:.4f}", True),
        ("Embedding capacity (bits)",
         "capacity_bits",  "{:,}",   True),
    ]

    all_pass = True
    for label, key, fmt, higher_is_better in metrics:
        b = baseline_metrics[key]
        f = finetuned_metrics[key]
        delta = f - b
        delta_str = ("+" if delta >= 0 else "") + fmt.format(delta)
        improved = delta > 0 if higher_is_better else delta < 0
        marker = "✓" if improved else "✗"
        print(f"  {marker} {label:<36} "
              f"{fmt.format(b):>10} "
              f"{fmt.format(f):>12} "
              f"{delta_str:>10}")
        if not improved:
            all_pass = False

    print(SEP2)

    print(f"  {'Mid-band ratio std (baseline)':<38} "
          f"{baseline_metrics['dct_mid_ratio_std']:>10.4f}")
    print(f"  {'Mid-band ratio std (fine-tuned)':<38} "
          f"{finetuned_metrics['dct_mid_ratio_std']:>10.4f}")
    print(f"  {'Images successfully embedded':<38} "
          f"{baseline_metrics['n_embedded']:>10} {finetuned_metrics['n_embedded']:>12}")

    print(SEP)
    print("  VERDICT")
    print(SEP2)

    base_ratio = baseline_metrics["dct_mid_ratio"]
    fine_ratio = finetuned_metrics["dct_mid_ratio"]
    dct_improved = fine_ratio > base_ratio
    psnr_ok      = finetuned_metrics["psnr_db"] > 38.0
    ssim_ok      = finetuned_metrics["ssim"] > 0.95
    bit_acc_ok   = finetuned_metrics["bit_accuracy"] > 0.99
    exact_ok     = finetuned_metrics["exact_recovery"] >= 0.99

    if dct_improved:
        pct = (fine_ratio - base_ratio) / max(base_ratio, 1e-8) * 100
        print(f"  ✓ DCT mid-band energy increased by {pct:.1f}% — fine-tuning WORKED")
    else:
        print(f"  ✗ DCT mid-band energy did NOT increase — fine-tuning had no effect")

    if psnr_ok:
        print(f"  ✓ PSNR {finetuned_metrics['psnr_db']:.2f} dB > 38 dB threshold — "
              f"embedding distortion acceptable")
    else:
        print(f"  ✗ PSNR {finetuned_metrics['psnr_db']:.2f} dB < 38 dB — "
              f"check K-stats or reduce SECRET_BITS")

    if ssim_ok:
        print(f"  ✓ SSIM {finetuned_metrics['ssim']:.4f} > 0.95 — "
              f"stego visually indistinguishable from cover")
    else:
        print(f"  ✗ SSIM {finetuned_metrics['ssim']:.4f} < 0.95 — "
              f"visible distortion after embedding")

    if bit_acc_ok:
        print(f"  ✓ Bit accuracy {finetuned_metrics['bit_accuracy']:.4f} — "
              f"payload recoverable")
    else:
        print(f"  ✗ Bit accuracy {finetuned_metrics['bit_accuracy']:.4f} — "
              f"DSTG round-trip unreliable on these covers")

    if exact_ok:
        print(f"  ✓ Exact recovery {finetuned_metrics['exact_recovery']:.2%} — "
              f"CRC-verified UTF-8 messages recoverable from PNG")
    else:
        print(f"  ✗ Exact recovery {finetuned_metrics['exact_recovery']:.2%} — "
              f"some PNGs failed full extraction")

    print(SEP)
    if dct_improved and psnr_ok and ssim_ok and bit_acc_ok and exact_ok:
        print("  OVERALL: ✓ PASS — thesis hypothesis supported")
        print("  Fine-tuning produced better steganographic carriers.")
    elif dct_improved:
        print("  OVERALL: ~ PARTIAL — DCT shift confirmed but DSTG quality")
        print("  needs attention. Inspect K_mean and per-image PSNR/SSIM.")
    else:
        print("  OVERALL: ✗ FAIL — consider more training steps or")
        print("  increasing lambda_dct (try --lambda_dct 0.5).")
    print(SEP + "\n")

    # ── Save raw metrics ───────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as fout:
        fout.write("BASELINE\n")
        for k, v in baseline_metrics.items():
            fout.write(f"  {k}: {v}\n")
        fout.write("\nFINETUNED\n")
        for k, v in finetuned_metrics.items():
            fout.write(f"  {k}: {v}\n")
    log.info(f"Raw metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()