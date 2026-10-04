"""
starvation.py — the neighbor-starvation width sweep.

Hypothesis: high-frequency detail can only develop through connected HR tokens, so a thin HR
region (width < ~2r, r = attention radius ~3) is starved and fails to grow its octave. Measure
octave energy INSIDE the HR tokens of each width-arm, divided by the SAME tokens in a uniform-HR
REF (same seed/trajectory, but every token has HR neighbors). REF is the perfect control.

Prediction: ratio ~ 1 for wide HR, falling off below HR width ~6-8 tokens (strip_w ~ 3-4 in
LR units, since rendered HR width = 2 x strip_w), ~0 at strip_w=1. Flat-to-1 everywhere => the
hypothesis is dead.

Generate first (same seed across all, E&R off, ALL arm):
  # REF = ~uniform HR (control)
  python -m flux.run_seam --device cuda:0 --prompts comm --group_mode off --only noer \
      --mask_pattern full --tag_prefix REF_
  # STRIPS: thin but full-width -> 1D-CONNECTED (HF can flow along the strip). partial starvation.
  for w in 8 4 2 1; do
    python -m flux.run_seam --device cuda:0 --prompts comm --group_mode off --only noer \
        --mask_pattern strips --strip_w $w --tag_prefix s${w}_
  done
  # SCATTERED: small blocks small in BOTH dims -> 2D-ISOLATED (no HR neighbours). true starvation.
  for b in 4 2 1; do
    python -m flux.run_seam --device cuda:0 --prompts comm --group_mode off --only noer \
        --mask_pattern scattered --strip_w $b --tag_prefix p${b}_
  done
  # fat-blob status quo (well-fed reference point)
  python -m flux.run_seam --device cuda:0 --prompts comm --group_mode off --only noer \
      --mask_pattern contiguous --tag_prefix blob_

Then (strips vs scattered at matched width tests CONNECTIVITY, not just neighbour count):
  python -m flux.starvation --root seam_out --ref REF_noer \
      --widths blob_noer,s8_noer,s4_noer,s2_noer,s1_noer,p4_noer,p2_noer,p1_noer
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def _gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def octave_map(g, lr_factor=2):
    """Per-pixel top-octave energy = the band a 2x-downsample discards (what HR adds over LR)."""
    lo = zoom(g, 1.0 / lr_factor, order=1)
    up = zoom(lo, (g.shape[0] / lo.shape[0], g.shape[1] / lo.shape[1]), order=1)
    return (g - up) ** 2


def _load_mask(path, shape):
    m = np.load(path)
    if m.shape != shape:
        m = zoom(m.astype(np.float32), (shape[0] / m.shape[0], shape[1] / m.shape[1]), order=0) > 0.5
    return m.astype(bool)


def analyze(root, ref_tag, width_tags):
    results = {w: [] for w in width_tags}      # tag -> list of (ratio, agree, wasted)
    for rd in sorted(glob.glob(os.path.join(root, ref_tag, "*"))):
        name = os.path.basename(rd)
        ref_png = os.path.join(rd, "mixed.png")
        if not os.path.exists(ref_png):
            continue
        oref = octave_map(_gray(ref_png))
        for w in width_tags:
            ad = os.path.join(root, w, name)
            arm_png, mpath = os.path.join(ad, "mixed.png"), os.path.join(ad, "mask.npy")
            if not (os.path.exists(arm_png) and os.path.exists(mpath)):
                continue
            arm = _gray(arm_png)
            hr = _load_mask(mpath, arm.shape)
            if hr.sum() < 16:
                continue
            oarm = octave_map(arm)
            a, b = oarm[hr], oref[hr]
            ratio = float(a.mean() / (b.mean() + 1e-9))                 # <1 = starved
            agree = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
            wasted = float((a < 0.5 * b).mean())                        # frac of HR tokens starved
            results[w].append((ratio, agree, wasted))
    return results


def main(args):
    tags = args.widths.split(",")
    res = analyze(args.root, args.ref, tags)
    print(f"root={args.root}  ref={args.ref}\n")
    print(f"{'arm':<14}{'octave_ratio':>13}{'agreement':>11}{'wasted_frac':>13}{'n':>4}")
    print("-" * 55)
    for w in tags:
        rows = np.array(res[w]) if res[w] else np.zeros((0, 3))
        if len(rows) == 0:
            print(f"{w:<14}{'(no data)':>13}")
            continue
        r, a, ws = np.nanmean(rows, axis=0)
        print(f"{w:<14}{r:>13.2f}{a:>11.2f}{ws:>13.2f}{len(rows):>4}")
    print("\noctave_ratio: octave inside HR tokens / same tokens in REF. ~1 = detail developed;")
    print("  << 1 = STARVED (tokens spent, no octave gain).  agreement: corr of octave with REF")
    print("  (low + low ratio = absent detail; higher = right-but-weak).  wasted_frac: share of")
    print("  HR tokens with <half the REF octave = the compute you paid for and didn't get.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seam_out")
    ap.add_argument("--ref", default="REF_noer")
    ap.add_argument("--widths", default="w8_noer,w4_noer,w2_noer,w1_noer,blob_noer")
    main(ap.parse_args())
