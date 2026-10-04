"""
trace.py — watch the HR region go wrong GRADUALLY over stage-2 steps, across many prompts.

Two trackers, both per stage-2 step:
  1. FRAMES: decode the MIXED x0 estimate (HR core + LR-upsampled fill) at every step, so you can
     see WHEN and WHERE the misalignment develops (clean at the promote step, wrong by the end).
  2. ATTENTION SPLIT: HR-query attention mass onto HR keys (itself) vs LR keys (the coarse anchor),
     per step. Tests two hypotheses at once:
       - "the HR region detaches from the LR anchor as it sharpens" -> LR-anchor fraction DROPS,
       - "HR over-attracts attention (mass inflation)" -> HR per-key vs LR per-key.

Runs a whole prompt category (like check_starv). Use the scattered problem mask to expose it.

    python -m flux.trace --device cuda:0 --category object_bisected --mask_pattern scattered --strip_w 2
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from . import transformer_flux as TF
from .run_seam import build_pipe, run_one
from . import comm_prompts
from .comm_prompts import PROMPTS


def _by_step(branch):
    """Aggregate one branch's attention records in TF._ATTN_LOG by stage-2 step (mean over blocks)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in TF._ATTN_LOG:
        if r.get("branch") != branch:
            continue
        st = r.get("step", -1)
        for k in ("HR_mass", "LR_mass", "HR_perkey", "LR_perkey", "entropy_norm", "HR_selfent"):
            if k in r:
                acc[st][k].append(r[k])
    return acc


def main(args):
    pipe = build_pipe(args)
    pipe.transformer.config.lr_rope_downsample_hr_tokens = True    # shipped grouping
    pipe.transformer.lr_group_mode = "mean"
    pipe.transformer.hr_lr_expand = False
    pipe.transformer.hr_lr_expand_masscal = False
    pipe.transformer.center_lr = False
    pipe.invert_hr_init = False
    pipe.custom_mask_pattern = None if args.mask_pattern == "saliency" else args.mask_pattern
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.custom_mask_strip_w = args.strip_w if args.strip_w > 0 else None
    pipe.transformer.attn_capture = bool(args.attn)
    TF._ATTN_MAX = 10 ** 7                                          # don't throttle over a full run

    os.makedirs(args.out_dir, exist_ok=True)
    if args.category == "all":                                     # every prompt, every category
        items = comm_prompts.flat()
    elif args.index >= 0:
        items = [(args.category, PROMPTS[args.category][args.index])]
    else:
        items = [(args.category, p) for p in PROMPTS[args.category]]
    print(f"tracing {len(items)} prompts (category={args.category}), mask={args.mask_pattern}"
          f"{'' if args.mask_pattern=='saliency' else f'(strip_w={args.strip_w})'}, {args.stg2_steps} stage-2 steps")

    hrq_anchor = defaultdict(list)     # HR query: fraction of mass on the LR anchor
    hrq_area = defaultdict(list)       # HR query: HR/LR mass-PER-AREA (= 4 x per-key ratio) = the count effect
    hrq_ent = defaultdict(list)        # HR query: FULL attention entropy (concentration)
    hrq_hh = defaultdict(list)         # HR query: HR->HR concentration (entropy within HR keys only)
    lrq_pull = defaultdict(list)       # LR query: fraction of mass pulled onto HR
    for pi, (cat, prompt) in enumerate(items):
        d = os.path.join(args.out_dir, f"{pi:02d}_{cat}"); os.makedirs(d, exist_ok=True)
        pipe._trace_steps = True; pipe._trace_dir = d; pipe._trace_tag = "x0"
        TF._ATTN_LOG.clear()
        run_one(pipe, args, prompt, 0)
        print(f"[{pi:02d} {cat}] {prompt[:40]}  -> {d}/x0_step*.png")
        if args.attn:
            for st, dd in _by_step("HRq").items():
                if st < 0:
                    continue
                hrm, lrm = np.mean(dd["HR_mass"]), np.mean(dd["LR_mass"])
                hrq_anchor[st].append(lrm / max(hrm + lrm, 1e-9))
                # mass-per-area ratio: HR key covers 1 HR-cell, lifted LR key covers 4 -> x4 on per-key
                hrq_area[st].append(4.0 * np.mean(dd["HR_perkey"]) / max(np.mean(dd["LR_perkey"]), 1e-9))
                if dd["entropy_norm"]:
                    hrq_ent[st].append(np.mean(dd["entropy_norm"]))
                if dd["HR_selfent"]:
                    hrq_hh[st].append(np.mean(dd["HR_selfent"]))
            for st, dd in _by_step("LRq").items():
                if st < 0:
                    continue
                hrm, lrm = np.mean(dd["HR_mass"]), np.mean(dd["LR_mass"])
                lrq_pull[st].append(hrm / max(hrm + lrm, 1e-9))

    if args.attn and hrq_anchor:
        print("\nAttention over stage-2 steps (mean over prompts):")
        print(" step | HRq LR-anchor frac | HRq HR/LR mass-per-area | HRq entropy | HR->HR entropy | LRq HR-pull frac")
        for st in sorted(hrq_anchor):
            print(f"  {st:>3} |       {np.mean(hrq_anchor[st]):.3f}        |         {np.mean(hrq_area[st]):>5.2f}x        "
                  f"|    {np.mean(hrq_ent.get(st,[np.nan])):.3f}    |     {np.mean(hrq_hh.get(st,[np.nan])):.3f}      "
                  f"|     {np.mean(lrq_pull.get(st,[np.nan])):.3f}")
        print("\nRead:")
        print("  HRq LR-anchor frac DROPS   -> HR region consults the coarse anchor less (bubble); note")
        print("     deep-HR queries have LR_mass~0 by geometry (too much HR around) -> anchor vanishes.")
        print("  HRq HR/LR mass-per-area    -> the COUNT effect: how much more HR attracts per unit AREA.")
        print("  HRq entropy vs HR->HR entropy -> FULL vs within-HR concentration. HR->HR isolates the")
        print("     'surrounded by too much HR' channel; compare to a uniform-1024 run for the null.")
        print("  LRq HR-pull frac RISES     -> LR queries pulled onto HR, starving LR regions ('steal').")
    print("\ndone ->", args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--stg1_steps", type=int, default=6)
    ap.add_argument("--stg2_steps", type=int, default=9)
    ap.add_argument("--lr_shift", type=float, default=1.8776105792643432)
    ap.add_argument("--hr_shift", type=float, default=3.0)
    ap.add_argument("--hr_ratio", type=float, default=0.6)
    ap.add_argument("--mask_type", choices=["base", "w_seg"], default="base")
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--boundary_grad", type=int, default=0)
    ap.add_argument("--invert_steps", type=int, default=4)
    ap.add_argument("--mask_pattern",
                    choices=["saliency", "bisecting", "contiguous", "corner", "strips", "scattered"],
                    default="scattered")
    ap.add_argument("--mask_hr_ratio", type=float, default=0.3)
    ap.add_argument("--strip_w", type=int, default=2)
    ap.add_argument("--attn", type=int, default=1, help="1 = also log the HR/LR attention split per step")
    ap.add_argument("--category", default="all",
                    help="'all' = every prompt in every category (like check_starv); or a category name")
    ap.add_argument("--index", type=int, default=-1, help="-1 = all prompts in the category")
    ap.add_argument("--out_dir", default="trace_out")
    main(ap.parse_args())
