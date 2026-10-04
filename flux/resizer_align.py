"""
resizer_align.py -- are the E&R latent resizers spatially aligned on REAL FLUX latents?

The native-context design (--native_ctx) makes proxy tokens with these resizers, so alignment between
the two resolutions stops depending on RoPE bookkeeping and starts depending on the resizers. This
checks them against true latents: encode an image at 1024 and a 2x-downsampled copy at 512 with the
FLUX VAE, then compare
    lt_down(L_1024) vs L_512        lt_up(L_512), bi_up(L_512) vs L_1024
Baselines (repeat-upsample / box-downsample) give the offset between the two encodes themselves;
a resizer is aligned if its shift matches the baseline. Shifts are in IMAGE PIXELS (dy, dx), with
>0 meaning content moved down/right. The top-left RoPE error we saw in mixed-res was ~8 px.

Synthetic latents are useless here: the resizers only reach ~0.5 correlation on them.

    python -m flux.resizer_align --device cuda:0            # finds fix_out/ or ../fix_out/ etc.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL

from .resizer_flux import instantiate_resizer


def shift(out, ref, frac=0.12):
    """Sub-pixel (dy, dx) with out(x) ~ ref(x - d): weighted LS on the low-frequency phase of the
    channel-summed cross-spectrum. Arrays [C, H, W]; result in pixels of that grid."""
    o = out - out.mean(axis=(1, 2), keepdims=True)
    r = ref - ref.mean(axis=(1, 2), keepdims=True)
    X = (np.fft.fft2(r) * np.conj(np.fft.fft2(o))).sum(0)
    H, W = X.shape
    u = np.fft.fftfreq(H)[:, None] * np.ones((1, W))
    v = np.ones((H, 1)) * np.fft.fftfreq(W)[None, :]
    sel = (np.abs(u) <= frac) & (np.abs(v) <= frac) & ~((u == 0) & (v == 0))
    w = np.sqrt(np.abs(X[sel]))
    A = np.stack([2 * np.pi * u[sel], 2 * np.pi * v[sel]], 1) * w[:, None]
    d, *_ = np.linalg.lstsq(A, np.angle(X[sel]) * w, rcond=None)
    return d


def corr(a, b):
    a = a - a.mean(axis=(1, 2), keepdims=True)
    b = b - b.mean(axis=(1, 2), keepdims=True)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@torch.no_grad()
def encode(vae, img, size, device, dtype):
    x = torch.from_numpy(np.asarray(img.resize((size, size), Image.BICUBIC, reducing_gap=None),
                                    dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    lat = vae.encode(x.to(device, dtype)).latent_dist.mode()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor          # transformer latent space


def main(args):
    dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(args.device)
    rz = {n: instantiate_resizer(os.path.join(args.ckpt_dir, f), k, "layer_norm", args.device, dtype).eval()
          for n, f, k in [("lt_up", "flux_lt_up.safetensors", "up"), ("bi_up", "flux_bi_up.safetensors", "up"),
                          ("lt_down", "flux_lt_down.safetensors", "down")]}
    pats = [args.images] if args.images else [
        "fix_out/*/*/mixed.png", "../fix_out/*/*/mixed.png", "scale_test/*/*/mixed.png",
        "../scale_test/*/*/mixed.png", "seam_out/*/*/mixed.png", "../seam_out/*/*/mixed.png"]
    paths = []
    for pat in pats:
        paths = sorted(glob.glob(pat))[: args.n]
        if paths:
            print(f"using {len(paths)} images from {pat}")
            break
    if not paths:
        raise SystemExit(f"no images found; tried {pats}. Pass --images 'path/*/mixed.png'")
    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        s = min(img.size); img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                                           (img.width + s) // 2, (img.height + s) // 2))
        L_hr, L_lr = encode(vae, img, 1024, args.device, dtype), encode(vae, img, 512, args.device, dtype)
        with torch.no_grad():
            outs = {
                "repeat_up (baseline)": (F.interpolate(L_lr, scale_factor=2, mode="nearest"), L_hr, 8),
                "lt_up": (rz["lt_up"](L_lr), L_hr, 8),
                "bi_up": (rz["bi_up"](L_lr), L_hr, 8),
                "box_down (baseline)": (F.avg_pool2d(L_hr, 2), L_lr, 16),
                "lt_down": (rz["lt_down"](L_hr), L_lr, 16),
            }
        row = {}
        for name, (o, r, px) in outs.items():
            o, r = o[0].float().cpu().numpy(), r[0].float().cpu().numpy()
            row[name] = (shift(o, r) * px, corr(o, r))
        rows.append(row)
        print(os.path.relpath(p), " ".join(f"{n.split()[0]}={v[0][0]:+.1f},{v[0][1]:+.1f}px(r={v[1]:.2f})"
                                           for n, v in row.items()))
    print("\nmean over images: shift (dy, dx) in image px   |  correlation with the true latent")
    for name in rows[0]:
        d = np.mean([r[name][0] for r in rows], 0); c = np.mean([r[name][1] for r in rows])
        print(f"  {name:<22} ({d[0]:+5.1f}, {d[1]:+5.1f}) px   r={c:.3f}")
    print("\nRead: a resizer is aligned if its shift matches its baseline (same direction). A consistent"
          "\n~8 px gap from the baseline would reproduce the top-left error inside the proxies.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--images", default="", help="glob; empty = search fix_out/scale_test/seam_out here and one level up")
    ap.add_argument("--n", type=int, default=12)
    main(ap.parse_args())
