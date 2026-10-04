"""
attn_batch.py — run the attention probe over ALL masks x prompts, saving per (mask, prompt):
  mixed.png              the generated image
  spatial_promote.png    HR-query attention map at stage-2 step 0 (the promote step)
  spatial_last.png       HR-query attention map at the last stage-2 step
  reach.txt              per-key 50%/75% physical reach for HRq/LRq at promote & last step

E&R off, ALL arm (group off), fixed seed. Captures are step-resolved (promote vs last).

    python -m flux.attn_batch --device cuda:0 --out_dir attn_out
    python -m flux.attn_batch --device cuda:0 --masks p2,p4,contiguous --n_prompts 6
"""

import argparse
import os

from . import transformer_flux as TF
from .run_seam import build_pipe, run_one
from . import comm_prompts
from .attn_probe import _save_spatial, _reach

# (tag, pattern, strip_w)
MASKS = [
    ("contiguous", "contiguous", 0), ("bisecting", "bisecting", 0),
    ("s8", "strips", 8), ("s4", "strips", 4), ("s2", "strips", 2), ("s1", "strips", 1),
    ("p4", "scattered", 4), ("p2", "scattered", 2), ("p1", "scattered", 1),
]


def _reset():
    TF._ATTN_LOG.clear(); TF._DIST_LOG.clear(); TF._DIST_CNT["n"] = 0
    TF._SPATIAL.clear(); TF._SPATIAL_STEPS_DONE.clear()


def _save_outputs(outdir):
    os.makedirs(outdir, exist_ok=True)
    sp = list(TF._SPATIAL)
    if sp:
        steps = sorted({s["step"] for s in sp})
        for st in steps:
            label = "promote" if st == 0 else ("last" if st == max(steps) else f"step{st}")
            _save_spatial([s for s in sp if s["step"] == st], os.path.join(outdir, f"spatial_{label}.png"))
    dl, binw = TF._DIST_LOG, TF._DIST_BIN_W
    with open(os.path.join(outdir, "reach.txt"), "w") as f:
        for key in sorted(dl):
            f.write(f"{key}: 50%-reach={_reach(dl[key], binw):.1f}  75%-reach={_reach(dl[key], binw, 0.75):.1f}\n")


def main(args):
    pipe = build_pipe(args)
    pipe.transformer.config.lr_rope_downsample_hr_tokens = False
    pipe.transformer.lr_group_mode = "off"
    pipe.transformer.hr_lr_expand = False
    pipe.transformer.hr_lr_expand_masscal = False
    pipe.invert_hr_init = False
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.transformer.attn_capture = True

    sel = set(args.masks.split(",")) if args.masks else None
    masks = [m for m in MASKS if (sel is None or m[0] in sel)]
    items = comm_prompts.flat()
    if args.n_prompts > 0:
        items = items[:args.n_prompts]

    for i, (cat, prompt) in enumerate(items):
        pname = f"{i:02d}_{cat}"
        for tag, pattern, sw in masks:
            pipe.custom_mask_pattern = pattern
            pipe.custom_mask_strip_w = sw if sw > 0 else None
            outdir = os.path.join(args.out_dir, tag, pname)
            print(f"[{tag}] {pname}: {prompt[:44]}")
            _reset()
            img = run_one(pipe, args, prompt, 0)          # pad=0 -> E&R off
            os.makedirs(outdir, exist_ok=True)
            img.save(os.path.join(outdir, "mixed.png"))
            _save_outputs(outdir)
    print("done ->", args.out_dir)


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
    ap.add_argument("--mask_hr_ratio", type=float, default=0.3)
    ap.add_argument("--boundary_grad", type=int, default=0)
    ap.add_argument("--invert_steps", type=int, default=4)
    ap.add_argument("--masks", default="", help="comma list of mask tags (default: all)")
    ap.add_argument("--n_prompts", type=int, default=-1, help="-1 = all comm prompts")
    ap.add_argument("--out_dir", default="attn_out")
    main(ap.parse_args())
