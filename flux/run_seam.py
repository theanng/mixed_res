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
from .seam_prompts import flat


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
    pipe.gaze_model = deepgaze_pytorch.DeepGazeI(pretrained=True).to(args.device).eval()
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
        mask_type=args.mask_type, pad1=pad, pad2=pad,
    ).images[0]


def main(args):
    pipe = build_pipe(args)
    settings = {"er": args.pad, "noer": 0}
    if args.only:
        settings = {args.only: settings[args.only]}
    for category, prompt in flat():
        idx = main.counter.setdefault(category, 0)
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
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--out_dir", default="./seam_out")
    main(ap.parse_args())
