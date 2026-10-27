"""
run_seam.py — run the 2-stage FLUX pipeline over the seam-exposing prompts, BOTH with and
without Boundary Expand-and-Replace, and save images + HR masks for seam_diagnosis.

E&R is gated by pad1/pad2 in the pipeline: (pad1>0 and pad2>0) => boundary exchange ON;
pad1=pad2=0 => OFF (the band masks aren't even built). Everything else (CRPA, segmentation
mask, learned resizers, partial inversion, feathered decode) stays identical, so this is a
clean E&R ablation.

    python -m flux.run_seam --device cuda:0
    python -m flux.run_seam --device cuda:0 --only er      # only with-E&R
    python -m flux.run_seam --device cuda:0 --mask_type base --hr_ratio 0.6

Outputs:
    seam_out/<er|noer>/<category>_<i>/mixed.png   the 1024 image
    seam_out/<er|noer>/<category>_<i>/mask.npy    full-res bool HR mask (True = HR)
    seam_out/<er|noer>/<category>_<i>/meta.json   {category, prompt}
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import deepgaze_pytorch
from scipy.ndimage import zoom
from scipy.special import logsumexp
from ultralytics import YOLOE
from diffusers import AutoencoderTiny

from .pipeline_flux_2stg import FluxPipeline
from .resizer_flux import instantiate_resizer
from .transformer_flux_2t import FluxTransformer2DModel
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from . import seam_prompts, comm_prompts


def build_pipe(args, dtype=torch.bfloat16):
    transformer = FluxTransformer2DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=dtype)
    pipe = FluxPipeline.from_pretrained(args.model_id, transformer=transformer, torch_dtype=dtype).to(args.device)

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", use_dynamic_shifting=False, shift=args.lr_shift)
    pipe.scheduler_hr = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", use_dynamic_shifting=False, shift=args.hr_shift)

    pipe.tvae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(args.device)

    gaze_shape = (args.height, args.width)
    cb = np.load(os.path.join(args.ckpt_dir, "centerbias_mit1003.npy"))
    cb = zoom(cb, (gaze_shape[0] / cb.shape[0], gaze_shape[1] / cb.shape[1]), order=0, mode="nearest")
    cb -= logsumexp(cb)
    pipe.centerbias_tensor = torch.from_numpy(cb).unsqueeze(0).to(args.device)
    pipe.gaze_shape = gaze_shape
    # NOTE: assign then .eval() on a SEPARATE line — DeepGaze's .eval() returns None, so
    # chaining `.to(device).eval()` would store None (matches the author's demo_2stg.py).
    pipe.gaze_model = deepgaze_pytorch.DeepGazeI(pretrained=True).to(args.device)
    pipe.gaze_model.eval()
    if args.mask_type == "w_seg":
        pipe.seg_model = YOLOE(os.path.join(args.ckpt_dir, "yoloe-26s-seg-pf.pt")).to(args.device)

    pipe.lt_upscaler = instantiate_resizer(os.path.join(args.ckpt_dir, "flux_lt_up.safetensors"),
                                           "up", "layer_norm", args.device, dtype)
    pipe.lt_downscaler = instantiate_resizer(os.path.join(args.ckpt_dir, "flux_lt_down.safetensors"),
                                             "down", "layer_norm", args.device, dtype)
    pipe.bi_upscaler = instantiate_resizer(os.path.join(args.ckpt_dir, "flux_bi_up.safetensors"),
                                           "up", "layer_norm", args.device, dtype)
    return pipe


def save_case(pipe, image, args, tag, category, i, prompt):
    d = os.path.join(args.out_dir, tag, f"{category}_{i:02d}")
    os.makedirs(d, exist_ok=True)
    image.save(os.path.join(d, "mixed.png"))
    # full-res bool HR mask (True = HR), from the LR-grid mask the pipeline just used
    fine = args.height * 2
    m = pipe._last_hr_mask_LR.float()                       # [b,1,ht,wt]
    m_full = F.interpolate(m, size=(fine, fine), mode="nearest")[0, 0] > 0.5
    np.save(os.path.join(d, "mask.npy"), m_full.cpu().numpy())
    json.dump(dict(category=category, prompt=prompt), open(os.path.join(d, "meta.json"), "w"))


def run_one(pipe, args, prompt, pad):
    return pipe(
        prompt, height=args.height, width=args.width, guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        hr_ratio=args.hr_ratio, stg1_true_steps=args.stg1_steps, stg2_true_steps=args.stg2_steps,
        mask_type=args.mask_type, pad1=pad, pad2=pad, boundary_grad=args.boundary_grad,
    ).images[0]


def main(args):
    pipe = build_pipe(args)
    # LR->HR grouping (down-path) mode:
    #   off     -> every LR query attends to every HR key (no downsample; slowest, raw CRPA)
    #   topleft -> original repo: post-norm top-left decimation of each 2x2 group
    #   mean    -> our pre-norm mean pool of each 2x2 group
    pipe.transformer.config.lr_rope_downsample_hr_tokens = (args.group_mode != "off")
    pipe.transformer.lr_group_mode = args.group_mode
    pipe.transformer.hr_lr_expand = args.expand                 # HR->LR footprint expansion (up-path)
    pipe.transformer.hr_lr_expand_masscal = args.expand_masscal  # -log(K) mass calibration
    pipe.transformer.attn_area_scale = args.area_scale          # area-weighted softmax (+log(area) on keys)
    pipe.native_ctx = pipe.transformer.native_ctx = args.native_ctx   # single-resolution canvases (proxy context)
    pipe.native_band = args.native_band                         # HR proxy band width (LR cells)
    pipe.relay_resampler = args.relay_resampler                 # relay: learned | bilinear
    pipe.relay_gain = args.relay_gain
    pipe.relay_cut = args.relay_cut
    pipe.relay_text_bias = args.relay_text_bias if args.relay_text_bias == "auto" else float(args.relay_text_bias)
    pipe.native_band_mode = args.native_band_mode               # lf = band keeps own detail; replace = plain E&R
    pipe.invert_hr_init = args.invert_hr        # structure-locked HR init (flow-inversion vs random)
    pipe.invert_hr_steps = args.invert_steps
    pipe.custom_mask_pattern = None if args.mask_pattern == "saliency" else args.mask_pattern
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.custom_mask_strip_w = args.strip_w if args.strip_w > 0 else None
    pipe.transformer.center_lr = args.center_lr    # +0.5 LR footprint-centroid phase fix
    if args.expand and args.center_lr:
        # expand places the 4 copies at anchor+{0,1}; with center_lr the anchor is 2y+1, so the
        # copies land on 2y+1, 2y+2 instead of the footprint 2y, 2y+1 (one HR cell off)
        raise SystemExit("--expand needs integer anchors: drop --center_lr")
    if args.lr_subcell and args.group_mode != "off":
        raise SystemExit("--lr_subcell needs --group_mode off (grouping keeps only one sub-token's position)")
    pipe.transformer.lr_subcell = args.lr_subcell  # HR keys at sub-cell centroids (y+-0.25) for LR queries
    pre = args.tag_prefix
    settings = {pre + "er": args.pad, pre + "noer": 0}
    if args.only:
        settings = {pre + args.only: settings[pre + args.only]}
    prompt_set = {"seam": seam_prompts, "comm": comm_prompts}[args.prompts]
    for category, prompt in prompt_set.flat():
        idx = main.counter.setdefault(category, 0)
        if args.cats and category not in args.cats:
            main.counter[category] += 1
            continue
        for tag, pad in settings.items():
            print(f"[{tag}] {category} #{idx}: {prompt[:50]}")
            img = run_one(pipe, args, prompt, pad)
            save_case(pipe, img, args, tag, category, idx, prompt)
        main.counter[category] += 1
    print("done ->", args.out_dir)


main.counter = {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=512)   # coarse; fine = 2x = 1024
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--stg1_steps", type=int, default=6)
    ap.add_argument("--stg2_steps", type=int, default=9)
    ap.add_argument("--lr_shift", type=float, default=1.8776105792643432)
    ap.add_argument("--hr_shift", type=float, default=3.0)
    ap.add_argument("--hr_ratio", type=float, default=0.6)
    ap.add_argument("--mask_type", choices=["base", "w_seg"], default="w_seg")
    ap.add_argument("--pad", type=int, default=2, help="E&R band size (the 'er' setting)")
    ap.add_argument("--only", choices=["er", "noer"], default=None)
    ap.add_argument("--prompts", choices=["seam", "comm"], default="seam",
                    help="seam=texture boundary prompts, comm=structural cross-scale-communication prompts")
    ap.add_argument("--mask_pattern", choices=["saliency", "full", "contiguous", "bisecting", "strips", "scattered", "salblock", "salcut", "salcheck", "saleyes"],
                    default="saliency", help="saliency=detector mask; full=uniform-HR REF; others=hand-written geometry")
    ap.add_argument("--mask_hr_ratio", type=float, default=0.3, help="HR area for hand-written masks (constant budget)")
    ap.add_argument("--strip_w", type=int, default=0, help="HR strip/block width in LR tokens for the width sweep (0=default)")
    ap.add_argument("--center_lr", action="store_true", help="+0.5 LR HR-grid coords (footprint-centroid phase fix)")
    ap.add_argument("--lr_subcell", action="store_true",
                    help="LR queries see each HR key at its own sub-cell centroid (y+-0.25, x+-0.25) instead of "
                         "4 keys stacked on the parent index. Needs --group_mode off; pair with --area_scale")
    ap.add_argument("--group_mode", choices=["off", "topleft", "mean"], default="mean",
                    help="LR->HR down-path: off=all HR keys, topleft=original decimation, mean=pre-norm pool")
    ap.add_argument("--expand", action="store_true", help="HR->LR footprint expansion for HR queries (up-path)")
    ap.add_argument("--expand_masscal", action="store_true", help="subtract log(4) from replicated LR logits")
    ap.add_argument("--area_scale", nargs="?", const="both", default="off", choices=["off", "hr", "lr", "both"],
                    help="area-weighted softmax (+log(area) on key logits). bare flag = both; "
                         "hr = HR-query branch only (LR-anchor +log4); lr = LR-query branch only")
    ap.add_argument("--native_ctx", choices=["off", "near", "full", "relay"], default="off",
                    help="single-resolution canvases with E&R-resizer proxies (forces the proxy refresh on, "
                         "overrides --pad). near = far field still via CRPA; full = no cross-resolution keys")
    ap.add_argument("--relay_resampler", choices=["learned", "bilinear"], default="learned",
                    help="relay: learned = E&R resizers (bi_up / lt_down); bilinear = training-free bilinear up + avg-pool down")
    ap.add_argument("--relay_gain", type=float, default=1.0,
                    help="relay: strength of the fake-LR correction passed to HR (0 = two streams, no relay)")
    ap.add_argument("--relay_cut", type=float, default=1.0,
                    help="relay: low-pass cutoff as a fraction of the LR Nyquist (1 = all an LR grid can hold)")
    ap.add_argument("--relay_text_bias", default="0",
                    help="relay: logit bias on text keys in the HR stream; 'auto' = log(HR tokens / full-image tokens)")
    ap.add_argument("--native_band", type=int, default=2, help="HR proxy band around the core, in LR cells")
    ap.add_argument("--native_band_mode", choices=["lf", "replace"], default="lf",
                    help="lf = band keeps its own high frequencies, only low frequencies locked to LR; "
                         "replace = plain E&R (band reset to blurry up(LR) each step)")
    ap.add_argument("--invert_hr", action="store_true", help="structure-locked HR init via flow-inversion (vs random re-noise)")
    ap.add_argument("--invert_steps", type=int, default=4, help="# Euler steps for the HR flow-inversion")
    ap.add_argument("--boundary_grad", type=int, default=0,
                    help="ramp width (HR/LR latent tokens) for graded boundary bands; 0 = hard step")
    ap.add_argument("--cats", nargs="+", default=None,
                    help="only these prompt categories, e.g. face_bisected periodic_bisected (indices stay stable)")
    ap.add_argument("--tag_prefix", default="", help='output subfolder prefix, e.g. "group_" -> group_er/group_noer')
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--out_dir", default="./seam_out")
    main(ap.parse_args())
