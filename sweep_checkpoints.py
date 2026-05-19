"""
sweep_checkpoints.py
─────────────────────────────────────────────────────────────────────────────
Sweep through training checkpoints and measure how DSTG embedding quality
evolves over the course of fine-tuning.

For each selected checkpoint:
  1. Load the LoRA-fine-tuned UNet
  2. Generate N images using identical seeds (so only the model changes)
  3. Run DSTG embed → PNG → reload → extract on each image
  4. Record K_mean, mid-band ratio, PSNR, SSIM, bit accuracy, exact recovery

Then print a table, save a CSV, and write a PNG plot showing how each metric
evolves with training step.

The vanilla (non-fine-tuned) SD 1.5 model is included as the leftmost data point
at step 0 for reference.

Usage
─────
    # Default: sweep every 6th checkpoint (8 points spanning the run)
    python sweep_checkpoints.py

    # More granular (slower):
    python sweep_checkpoints.py --stride 3 --num_images 6

    # Specific checkpoints by step number:
    python sweep_checkpoints.py --steps 2000,10000,30000,80000
"""

import argparse
import gc
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import everything from validate.py — no duplication ───────────────────────
import validate as v

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_ROOT = Path("./finetune_output")
SWEEP_OUTPUT    = Path("./validation_output/sweep")
DEFAULT_STRIDE  = 6        # 80000/2000=40 ckpts; stride 6 = 7 ckpts + final + baseline
DEFAULT_IMAGES  = 4        # per checkpoint (small for speed; raise for stable numbers)
DEFAULT_SEED    = 42
# ──────────────────────────────────────────────────────────────────────────────


def discover_checkpoints(root: Path) -> list:
    """
    Find all checkpoint_stepNNNNNNN directories under root and return them
    as a sorted list of (step_number, path) tuples.  Also includes the 'final'
    directory if it exists, tagged with the highest step number found + 1.
    """
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root.resolve()}")

    pattern = re.compile(r"checkpoint_step0*(\d+)$")
    checkpoints = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m and (child / "unet.pt").exists():
            checkpoints.append((int(m.group(1)), child))

    checkpoints.sort(key=lambda x: x[0])

    # Include 'final' as a separate entry if present and not duplicating last step
    final_dir = root / "final"
    if final_dir.exists() and (final_dir / "unet.pt").exists():
        last_step = checkpoints[-1][0] if checkpoints else 0
        checkpoints.append((last_step, final_dir))   # treat 'final' as last-step

    return checkpoints


def select_checkpoints(all_ckpts: list, stride: int = None,
                        steps: list = None) -> list:
    """
    Pick which checkpoints to evaluate.
    If steps is given, return only checkpoints whose step is in that list.
    Otherwise sample every <stride>th, always including the first and last.
    """
    if steps:
        steps_set = set(steps)
        return [(s, p) for s, p in all_ckpts if s in steps_set]

    if stride is None or stride < 1:
        return all_ckpts

    if len(all_ckpts) == 0:
        return []

    # Take every `stride`-th, then ensure first and last are included
    selected = all_ckpts[::stride]
    if all_ckpts[-1] not in selected:
        selected.append(all_ckpts[-1])
    return selected


def evaluate_one_model(label: str, finetuned: bool, finetuned_dir: Path,
                        num_images: int, seed: int, device, dtype,
                        output_root: Path) -> dict:
    """
    Load a model, generate `num_images` images, evaluate them, and return
    the metrics dictionary.  Frees GPU memory before returning.
    """
    log.info(f"━━━━ {label} ━━━━")
    t0 = time.time()

    tok, enc, vae, unet, sched = v.load_pipeline(
        device, dtype, finetuned=finetuned, finetuned_dir=finetuned_dir)

    images = v.generate_images(
        tok, enc, vae, unet, sched, device, dtype,
        num_images=num_images, seed=seed)

    # Free model memory before doing the (CPU-bound) DSTG eval
    del tok, enc, vae, unet, sched
    gc.collect()
    torch.cuda.empty_cache()

    metrics = v.evaluate_images(images, label, output_root)
    metrics["wall_seconds"] = time.time() - t0
    metrics["label"]        = label

    log.info(f"  done in {metrics['wall_seconds']:.1f}s  "
             f"K={metrics['K_mean']:.1f}  "
             f"ratio={metrics['dct_mid_ratio']:.3f}  "
             f"bit_acc={metrics['bit_accuracy']:.3f}  "
             f"exact={metrics['exact_recovery']:.2f}")
    return metrics


def write_csv(rows: list, path: Path):
    """Write metrics rows as a CSV."""
    if not rows:
        return
    cols = ["step", "label", "K_mean", "dct_mid_ratio", "psnr_db", "ssim",
            "bit_accuracy", "exact_recovery", "capacity_bits", "n_embedded",
            "wall_seconds"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    log.info(f"CSV saved → {path}")


def write_plot(rows: list, path: Path):
    """
    Save a 3-panel plot:  K vs step, mid-band ratio vs step, bit accuracy vs step.
    """
    steps   = [r["step"] for r in rows]
    K       = [r["K_mean"] for r in rows]
    ratio   = [r["dct_mid_ratio"] for r in rows]
    bit_acc = [r["bit_accuracy"] for r in rows]
    exact   = [r["exact_recovery"] for r in rows]
    psnr    = [r["psnr_db"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, K, "o-", color="steelblue")
    axes[0, 0].axhline(17.3, ls="--", color="grey", alpha=0.6,
                       label="baseline K=17.3 (previous run)")
    axes[0, 0].set_title("DSTG K_mean vs training step")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("K (eligible coeffs / block)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(steps, ratio, "o-", color="darkorange")
    axes[0, 1].set_title("Mid-band DCT energy ratio vs training step")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("ratio (mid energy / total)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, bit_acc, "o-", color="seagreen", label="bit accuracy")
    axes[1, 0].plot(steps, exact,   "s--", color="firebrick", label="exact recovery")
    axes[1, 0].axhline(0.99, ls="--", color="grey", alpha=0.6, label="0.99 target")
    axes[1, 0].set_title("DSTG bit accuracy & exact recovery vs training step")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("rate (0..1)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, psnr, "o-", color="purple")
    axes[1, 1].axhline(38.0, ls="--", color="grey", alpha=0.6, label="38 dB target")
    axes[1, 1].set_title("PSNR cover→stego vs training step")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log.info(f"Plot saved → {path}")


def print_table(rows: list):
    """Print a clean comparison table to stdout."""
    SEP = "─" * 88
    print()
    print("═" * 88)
    print("  CHECKPOINT SWEEP RESULTS")
    print("═" * 88)
    print(f"  {'step':>7}  {'label':<14} {'K':>6} {'ratio':>7} "
          f"{'PSNR':>7} {'SSIM':>7} {'bit_acc':>8} {'exact':>7} "
          f"{'cap_bits':>10}")
    print(SEP)
    for r in rows:
        print(f"  {r['step']:>7}  {r['label']:<14} "
              f"{r['K_mean']:>6.1f} {r['dct_mid_ratio']:>7.3f} "
              f"{r['psnr_db']:>7.2f} {r['ssim']:>7.4f} "
              f"{r['bit_accuracy']:>8.3f} {r['exact_recovery']:>7.2f} "
              f"{r['capacity_bits']:>10,}")
    print(SEP)

    # Pick the "best" checkpoint by a simple criterion:
    # exact_recovery first, then bit_accuracy, then K_mean
    scored = [(r["exact_recovery"], r["bit_accuracy"], r["K_mean"], r)
              for r in rows if r["label"] != "baseline"]
    if scored:
        scored.sort(reverse=True)
        best = scored[0][-1]
        print(f"  Best fine-tuned checkpoint by reliability:  step {best['step']}  "
              f"(exact={best['exact_recovery']:.2f}, bit_acc={best['bit_accuracy']:.3f}, "
              f"K={best['K_mean']:.1f})")
        print("═" * 88 + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", type=str, default=str(CHECKPOINT_ROOT),
                   help="Directory containing checkpoint_stepNNNNNNN/ folders")
    p.add_argument("--sweep_output",    type=str, default=str(SWEEP_OUTPUT),
                   help="Where to write images, CSV, and plot")
    p.add_argument("--num_images",      type=int, default=DEFAULT_IMAGES,
                   help="Images to generate per checkpoint (default 4)")
    p.add_argument("--stride",          type=int, default=DEFAULT_STRIDE,
                   help="Evaluate every Nth checkpoint (ignored if --steps given)")
    p.add_argument("--steps",           type=str, default=None,
                   help="Comma-separated step numbers to evaluate, e.g. '2000,10000,80000'")
    p.add_argument("--seed",            type=int, default=DEFAULT_SEED)
    p.add_argument("--skip_baseline",   action="store_true",
                   help="Don't evaluate the vanilla SD model as a reference")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    sweep_output = Path(args.sweep_output)
    sweep_output.mkdir(parents=True, exist_ok=True)

    if device.type != "cuda":
        log.warning("CUDA not available — this script will be extremely slow on CPU.")

    # ── Discover and select checkpoints ────────────────────────────────────
    all_ckpts = discover_checkpoints(Path(args.checkpoint_root))
    if not all_ckpts:
        log.error(f"No checkpoints found in {args.checkpoint_root}")
        sys.exit(1)
    log.info(f"Found {len(all_ckpts)} checkpoints in {args.checkpoint_root}")

    steps_filter = None
    if args.steps:
        steps_filter = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    selected = select_checkpoints(all_ckpts, stride=args.stride, steps=steps_filter)

    if not selected:
        log.error("No checkpoints matched the selection criteria.")
        sys.exit(1)
    log.info(f"Evaluating {len(selected)} checkpoints: "
             f"{[s for s, _ in selected]}")
    log.info(f"Images per checkpoint: {args.num_images}  "
             f"(estimated total time: ~{len(selected) * args.num_images * 8 / 60:.0f} min)")

    rows = []

    # ── Optional baseline (vanilla SD 1.5) ────────────────────────────────
    if not args.skip_baseline:
        m = evaluate_one_model(
            label         = "baseline",
            finetuned     = False,
            finetuned_dir = Path(args.checkpoint_root),   # unused for vanilla
            num_images    = args.num_images,
            seed          = args.seed,
            device        = device,
            dtype         = dtype,
            output_root   = sweep_output,
        )
        m["step"] = 0
        rows.append(m)

    # ── Sweep through fine-tuned checkpoints ──────────────────────────────
    for step_num, ckpt_dir in selected:
        label = f"step_{step_num:07d}"
        try:
            m = evaluate_one_model(
                label         = label,
                finetuned     = True,
                finetuned_dir = ckpt_dir,
                num_images    = args.num_images,
                seed          = args.seed,
                device        = device,
                dtype         = dtype,
                output_root   = sweep_output,
            )
        except Exception as e:
            log.error(f"Checkpoint {label} failed: {e}")
            continue
        m["step"] = step_num
        rows.append(m)

    # ── Outputs ────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["step"])

    print_table(rows)
    write_csv(rows, sweep_output / "sweep_metrics.csv")
    write_plot(rows, sweep_output / "sweep_plot.png")

    log.info(f"Per-checkpoint images saved under {sweep_output}/<label>/")
    log.info(f"Open {sweep_output / 'sweep_plot.png'} to see the curves.")


if __name__ == "__main__":
    main()