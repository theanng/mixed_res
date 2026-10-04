"""
size_check.py — is the LR<->HR object drift a LATENT/NOISE effect, not an embedding one?

The latent geometry is top-left/row-major consistent (pack <-> RoPE <-> upsample all verified in
code), so "dividing into HR tokens" does NOT shift or rescale the structure. The concrete suspect
is the noise re-injection: the HR region is re-noised to a HIGH stage-2 start sigma (~0.8 with
hr_shift=3) on top of the BLURRY upsampled-LR x0, with FRESH noise -> it substantially re-generates
and can drift from the LR structure.

This runs each prompt three ways and saves, so you can eyeball the eye across them:
  p{i}_hr_init_x0.png   the upsampled-LR structure the HR region STARTS from (before HR denoising)
  p{i}_final_fresh.png  final image, default FRESH-noise HR init
  p{i}_final_lock.png   final image, STRUCTURE-LOCKED HR init (flow-inversion, less noise freedom)

Read: if the eye in final_fresh has DRIFTED from hr_init_x0 (size/shape), but final_lock stays
close to the init, the drift is the high-sigma fresh-noise re-generation -> a latent/noise cause,
fixable at the init (not embedding). The printed sigma says how much noise buries the structure.

    python -m flux.size_check --device cuda:0 --index -1
"""

import argparse
import os

import torch

from .run_seam import build_pipe, run_one
from .comm_prompts import PROMPTS


def _real_pipeline_defaults(pipe, args):
    pipe.transformer.config.lr_rope_downsample_hr_tokens = True   # the shipped grouping
    pipe.transformer.lr_group_mode = "mean"
    pipe.transformer.hr_lr_expand = False
    pipe.transformer.hr_lr_expand_masscal = False
    pipe.transformer.center_lr = False
    # mask: 'saliency' = CRPA's DeepGaze mask (HR on the whole subject -> LR/HR SEPARATED).
    # a pattern (e.g. 'bisecting') cuts THROUGH the subject so an object straddles the LR/HR seam.
    pipe.custom_mask_pattern = None if args.mask_pattern == "saliency" else args.mask_pattern
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.custom_mask_strip_w = args.strip_w if args.strip_w > 0 else None
    pipe.invert_hr_steps = args.invert_steps


def main(args):
    pipe = build_pipe(args)
    _real_pipeline_defaults(pipe, args)
    print(f"mask: {args.mask_pattern}"
          f"{'' if args.mask_pattern=='saliency' else f' (cuts through the subject -> object across the LR/HR seam)'}")
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = [PROMPTS[args.category][args.index]] if args.index >= 0 else PROMPTS[args.category]

    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi}] {prompt[:60]}")
        # (1) FRESH-noise init + dump the HR structure the region starts from
        pipe.invert_hr_init = False
        pipe._dump_hr_init = True
        pipe._dump_hr_init_path = os.path.join(args.out_dir, f"p{pi}_hr_init_x0.png")
        img = run_one(pipe, args, prompt, 0)
        img.save(os.path.join(args.out_dir, f"p{pi}_final_fresh.png"))

        # (2) STRUCTURE-LOCKED init (flow-inversion): the noise is correlated with the structure,
        # so the HR region refines rather than re-generates -> should drift LESS if noise is the cause
        pipe._dump_hr_init = False
        pipe.invert_hr_init = True
        img2 = run_one(pipe, args, prompt, 0)
        img2.save(os.path.join(args.out_dir, f"p{pi}_final_lock.png"))
        pipe.invert_hr_init = False

    print("\ndone ->", args.out_dir,
          "\ncompare per prompt: hr_init_x0 (structure) vs final_fresh (drift?) vs final_lock (held?)")


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
                    default="saliency",
                    help="saliency=CRPA DeepGaze (LR/HR separated); bisecting=seam through subject; "
                         "scattered=small isolated HR patches (the p2/p4 starvation case, use --strip_w)")
    ap.add_argument("--mask_hr_ratio", type=float, default=0.5, help="HR area for hand-written patterns")
    ap.add_argument("--strip_w", type=int, default=0, help="strip width for --mask_pattern strips")
    ap.add_argument("--category", default="face_bisected")
    ap.add_argument("--index", type=int, default=-1, help="-1 = all prompts in the category")
    ap.add_argument("--out_dir", default="size_out")
    main(ap.parse_args())
