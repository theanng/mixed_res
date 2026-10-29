"""
scale_prior.py -- Phase Q / X1: does FLUX.1-dev draw texture at a fixed size in NATIVE PIXELS
(a resolution prior, H1), or at a fixed fraction of the image (composition-invariant)?

Stock diffusers FluxPipeline, no mixed-resolution code. Each periodic prompt (comm periodic_bisected 5 +
seam periodic 7) at 512 and at 1024, seeds 0-3, default sampler. The dominant texture period is measured on
the central 50% crop (row-profile autocorrelation, periodicity >= 0.4; whole crop and its 4 quadrants, median).

    px ratio = period@1024 / period@512 (both in native pixels)
        ~1  -> H1: fixed native-pixel size. A full 1024 generation also has the "mini-bricks"; the HR region of a
               mixed run is behaving like normal FLUX@1024, and the LR region is what differs.
        ~2  -> composition-invariant: same fraction of the image at both sizes. The mini-bricks are made by the
               mixed pipeline (the HR region misreads its context), and a coupling fix is possible.

    python -m flux.scale_prior --device cuda:0 --out_dir ./phaseQ_out/x1            # ~96 images
    python -m flux.scale_prior --device cuda:0 --seeds 0 1 --steps 20               # quicker
Images are cached, so a rerun only measures (--measure_only never loads the model).
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from . import comm_prompts, seam_prompts
from .scr_metrics import luma, row_period


def prompts():
    out = [("periodic_bisected", i, p) for i, p in enumerate(comm_prompts.PROMPTS["periodic_bisected"])]
    out += [("periodic", i, p) for i, p in enumerate(seam_prompts.PROMPTS["periodic"])]
    return out


def measure(img):
    """Dominant row / column period (native px) on the central 50% crop: whole crop and 4 quadrants."""
    y = luma(np.asarray(img.convert("RGB"), dtype=np.float64))
    H, W = y.shape
    c = y[H // 4:3 * H // 4, W // 4:3 * W // 4]
    h, w = c.shape
    pieces = [c, c[:h // 2, :w // 2], c[:h // 2, w // 2:], c[h // 2:, :w // 2], c[h // 2:, w // 2:]]
    rows = [row_period(p) for p in pieces]
    cols = [row_period(p.T) for p in pieces]
    rows = [r for r in rows if r]
    cols = [r for r in cols if r]
    return {"row": float(np.median([r[0] for r in rows])) if rows else None,
            "row_corr": float(np.median([r[1] for r in rows])) if rows else None,
            "col": float(np.median([r[0] for r in cols])) if cols else None,
            "H": H}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = None
    res = []
    for cat, i, prompt in prompts():
        for size in (512, 1024):
            for seed in args.seeds:
                path = os.path.join(args.out_dir, str(size), f"{cat}_{i:02d}_s{seed}.png")
                if not os.path.exists(path):
                    if args.measure_only:
                        continue
                    if pipe is None:
                        import torch
                        from diffusers import FluxPipeline
                        pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)
                    import torch
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    img = pipe(prompt, height=size, width=size, guidance_scale=args.guidance_scale,
                               num_inference_steps=args.steps,
                               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
                    img.save(path)
                    print(f"[{size}] {cat}_{i:02d} seed {seed}")
                m = measure(Image.open(path))
                res.append(dict(cat=cat, i=i, prompt=prompt, size=size, seed=seed, **m))
    json.dump(res, open(os.path.join(args.out_dir, "scale_prior.json"), "w"), indent=1)

    med = lambda xs: float(np.median(xs)) if xs else None
    print("\nX1 -- texture period, native px (median over seeds with periodicity >= 0.4)")
    print(f"{'prompt':<42}{'row@512':>9}{'row@1024':>10}{'px ratio':>10}{'col@512':>9}{'col@1024':>10}{'px ratio':>10}")
    ratios = []
    for cat, i, prompt in prompts():
        r = [x for x in res if x["cat"] == cat and x["i"] == i]
        g = lambda size, k: med([x[k] for x in r if x["size"] == size and x[k] is not None])
        r5, r1, c5, c1 = g(512, "row"), g(1024, "row"), g(512, "col"), g(1024, "col")
        rr = r1 / r5 if r5 and r1 else None
        cr = c1 / c5 if c5 and c1 else None
        if rr:
            ratios.append(rr)
        f = lambda v, n=1: "     -" if v is None else f"{v:.{n}f}"
        print(f"{prompt[:41]:<42}{f(r5):>9}{f(r1):>10}{f(rr, 2):>10}{f(c5):>9}{f(c1):>10}{f(cr, 2):>10}")
    if ratios:
        m = float(np.median(ratios))
        verdict = ("H1: fixed native-pixel size (resolution prior)" if m < 1.4 else
                   "composition-invariant (same fraction of the image)" if m > 1.7 else "in between")
        print(f"\nmedian row px ratio 1024/512 = {m:.2f}  ->  {verdict}")
        print("reference: plain CRPA / E&R / C put HR bricks at 0.40x the LR bricks in the final 1024 canvas, "
              "i.e. ~0.8x in native px.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--out_dir", default="./phaseQ_out/x1")
    ap.add_argument("--measure_only", action="store_true", help="only measure images already on disk (no model)")
    main(ap.parse_args())
