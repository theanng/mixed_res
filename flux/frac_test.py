"""
frac_test.py -- do non-integer RoPE offsets break FLUX by themselves?

Single resolution, none of the mixed-res machinery: stock diffusers FluxPipeline at 1024. Half of
the image tokens (checkerboard) get their RoPE position moved by +s in both axes. Pairs inside each
half keep integer offsets, and every cross pair gets integer+s. This is HARSHER than --lr_subcell:
here the positions are physically wrong (the token pretends to sit s cells away), while lr_subcell's
fractional positions are the true footprint centroids.

    s = 0      baseline
    s = 0.25   quarter cell (lr_subcell's offset, in LR units)
    s = 0.5    half cell (center_lr's cross-resolution offset in HR units; RALU / PI-LR use it too)
    s = 0.75
    s = 1.0    integer offset: the same kind of displacement, but on the training lattice

Predictions
  "non-integer index breaks the model" -> 0.25 / 0.5 blur or collapse, 1.0 (integer) is mildest
  bandlimited RoPE (all FLUX spatial freqs <= 1 rad/step < pi) -> damage grows smoothly with s,
                                          1.0 is worst, 0.25 is close to baseline
Pattern 'all' moves every image token by +s: image-image offsets stay integer, only text-image
offsets turn fractional (FLUX puts text tokens at position 0). Expect ~baseline.

    python -m flux.frac_test --device cuda:0
    python -m flux.frac_test --device cuda:0 --size 512 --steps 20      # quicker
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline

from . import comm_prompts


class Offset:
    pattern = "checker"
    s = 0.0
    calls = 0


def install(pipe, off):
    """Wrap transformer.forward so img_ids get the offset (kept in float32; bf16 ids would round)."""
    orig = pipe.transformer.forward

    def fwd(*args, **kw):
        if off.s != 0.0:
            ids = kw["img_ids"]
            is3 = ids.dim() == 3
            f = (ids[0] if is3 else ids).float().clone()
            if off.pattern == "checker":
                m = (f[:, 1] + f[:, 2]).round().long() % 2 == 1
            else:
                m = torch.ones_like(f[:, 0], dtype=torch.bool)
            f[m, 1] += off.s
            f[m, 2] += off.s
            kw["img_ids"] = f[None] if is3 else f
            kw["txt_ids"] = kw["txt_ids"].float()
            off.calls += 1
        return orig(*args, **kw)

    pipe.transformer.forward = fwd


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def sharpness(g):
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return lap.var()


def psnr(a, b):
    mse = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))


def main(args):
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)
    off = Offset()
    install(pipe, off)

    cats = ["periodic_bisected", "text", "face_bisected", "object_bisected"]
    prompts = [(c, comm_prompts.PROMPTS[c][0]) for c in cats if c in comm_prompts.PROMPTS][: args.n]
    configs = [("checker", s) for s in args.checker] + [("all", s) for s in args.all]

    rows = {cfg: [] for cfg in configs}
    for cat, prompt in prompts:
        imgs = {}
        for pattern, s in [("base", 0.0)] + configs:
            off.pattern, off.s, off.calls = pattern, s, 0
            img = pipe(prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                       num_inference_steps=args.steps,
                       generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
            if s != 0.0 and off.calls == 0:
                raise RuntimeError("offset was never applied: transformer was not called with img_ids=")
            d = os.path.join(args.out_dir, f"{pattern}_s{s:g}")
            os.makedirs(d, exist_ok=True)
            img.save(os.path.join(d, f"{cat}.png"))
            imgs[(pattern, s)] = img
        base = imgs[("base", 0.0)]
        sb = sharpness(gray(base))
        for cfg in configs:
            rows[cfg].append((psnr(imgs[cfg], base), sharpness(gray(imgs[cfg])) / sb))
        # one strip per prompt: base | checker s... | all s...
        th = args.size // 2
        strip = Image.new("RGB", (th * (len(configs) + 1), th))
        for j, key in enumerate([("base", 0.0)] + configs):
            strip.paste(imgs[key].resize((th, th)), (j * th, 0))
        strip.save(os.path.join(args.out_dir, f"strip_{cat}.png"))
        print(f"[{cat}] done")

    print("\nvs baseline (mean over prompts)   PSNR dB (higher = closer)   sharpness ratio (1 = same, <1 = blur)")
    for cfg, r in rows.items():
        p, sh = np.mean([x[0] for x in r]), np.mean([x[1] for x in r])
        print(f"  {cfg[0]:<8} s={cfg[1]:<5g}  PSNR {p:6.2f}   sharp {sh:5.3f}")
    print("\nstrips (left to right): base, " + ", ".join(f"{p} s={s:g}" for p, s in configs))
    print("done ->", args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--n", type=int, default=4, help="number of prompts (one per category)")
    ap.add_argument("--checker", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--all", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--out_dir", default="./frac_out")
    main(ap.parse_args())
