"""
resizer_align.py -- how well does each LR<->HR resampler reproduce the TRUE other-resolution latent?

Ground truth: encode the same image at 1024 (L_hr) and at 512 (L_lr) with the FLUX VAE. Then map
L_lr -> 1024 and L_hr -> 512 with every candidate resampler and compare against the true latent.

Resamplers
  up   : repeat (nearest), bilinear / bicubic in LATENT space (training-free, ~free),
         lt_up / bi_up (CRPA's learned E&R resizers),
         pix_vae / pix_taef1 (decode -> bicubic x2 in PIXEL space -> re-encode; training-free,
         full FLUX VAE or the tiny taef1 autoencoder)
  down : box (avg-pool), lt_down (learned), pix_vae / pix_taef1 (decode -> area /2 -> re-encode)

Metrics (per image, then averaged)
  shift (dy, dx) in IMAGE PIXELS of the low-frequency phase; >0 = content moved down/right
  r_all : correlation with the true latent
  r_lo  : correlation of the part an LR grid can represent (ideal low-pass at the LR Nyquist,
          0.25 cyc / HR-latent px) -- what an "LF lock" (HR follows LR at low frequency) needs
  r_hi  : correlation of the rest (up only) -- the detail the HR stream has to invent itself
  ms    : time per call on this device

    python -m flux.resizer_align --device cuda:0                 # finds fix_out/ or ../fix_out/ etc.
    python -m flux.resizer_align --device cuda:0 --no_pixel      # skip the VAE round-trips
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, AutoencoderTiny

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
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def split_bands(x, cut=0.25):
    """Ideal low/high split of [C,H,W] at `cut` cycles/px (both axes). LR Nyquist on the HR grid = 0.25."""
    X = np.fft.fft2(x)
    H, W = x.shape[-2:]
    u = np.abs(np.fft.fftfreq(H))[:, None]
    v = np.abs(np.fft.fftfreq(W))[None, :]
    lo = (u < cut) & (v < cut)
    return np.real(np.fft.ifft2(X * lo)), np.real(np.fft.ifft2(X * ~lo))


def to_latent(vae, x):
    lat = vae.encode(x).latent_dist.mode()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor          # transformer latent space


def from_latent(vae, lat):
    return vae.decode(lat / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False)[0]


@torch.no_grad()
def encode(vae, img, size, device, dtype):
    x = torch.from_numpy(np.asarray(img.resize((size, size), Image.BICUBIC, reducing_gap=None),
                                    dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    return to_latent(vae, x.to(device, dtype))


def timed(fn, x, device):
    if "cuda" in str(device):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = fn(x)
    if "cuda" in str(device):
        torch.cuda.synchronize()
    return y, (time.perf_counter() - t0) * 1e3


def build_methods(args, vae, dtype):
    rz = {n: instantiate_resizer(os.path.join(args.ckpt_dir, f), k, "layer_norm", args.device, dtype).eval()
          for n, f, k in [("lt_up", "flux_lt_up.safetensors", "up"), ("bi_up", "flux_bi_up.safetensors", "up"),
                          ("lt_down", "flux_lt_down.safetensors", "down")]}
    up = {
        "repeat": lambda L: F.interpolate(L, scale_factor=2, mode="nearest"),
        "bilinear (latent)": lambda L: F.interpolate(L, scale_factor=2, mode="bilinear", align_corners=False),
        "bicubic (latent)": lambda L: F.interpolate(L, scale_factor=2, mode="bicubic", align_corners=False),
        "lt_up (learned)": rz["lt_up"],
        "bi_up (learned)": rz["bi_up"],
    }
    down = {
        "box (avg-pool)": lambda L: F.avg_pool2d(L, 2),
        "lt_down (learned)": rz["lt_down"],
    }
    if not args.no_pixel:
        tae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(args.device).eval()
        up_px = lambda img: F.interpolate(img, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)
        dn_px = lambda img: F.interpolate(img, scale_factor=0.5, mode="area")
        up["pix_vae"] = lambda L: to_latent(vae, up_px(from_latent(vae, L)))
        down["pix_vae"] = lambda L: to_latent(vae, dn_px(from_latent(vae, L)))
        up["pix_taef1"] = lambda L: tae.encode(up_px(tae.decode(L).sample)).latents
        down["pix_taef1"] = lambda L: tae.encode(dn_px(tae.decode(L).sample)).latents
    return up, down


def main(args):
    dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(args.device)
    up, down = build_methods(args, vae, dtype)
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

    rows = {}
    for k, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        s = min(img.size); img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                                           (img.width + s) // 2, (img.height + s) // 2))
        L_hr, L_lr = encode(vae, img, 1024, args.device, dtype), encode(vae, img, 512, args.device, dtype)
        ref_hr = L_hr[0].float().cpu().numpy(); ref_lr = L_lr[0].float().cpu().numpy()
        ref_lo, ref_hi = split_bands(ref_hr)
        for name, fn in up.items():
            o, ms = timed(fn, L_lr, args.device)
            o = o[0].float().cpu().numpy()
            o_lo, o_hi = split_bands(o)
            rows.setdefault(("up", name), []).append(
                (shift(o, ref_hr) * 8, corr(o, ref_hr), corr(o_lo, ref_lo), corr(o_hi, ref_hi), ms))
        for name, fn in down.items():
            o, ms = timed(fn, L_hr, args.device)
            o = o[0].float().cpu().numpy()
            rows.setdefault(("down", name), []).append(
                (shift(o, ref_lr) * 16, corr(o, ref_lr), float("nan"), float("nan"), ms))
        print(f"[{k + 1}/{len(paths)}] {os.path.relpath(p)}")

    print("\nmean over images   (shift in image px; r_lo = part an LR grid can represent, r_hi = the rest)")
    print(f"  {'':<5}{'method':<20}{'shift (dy, dx)':>18}{'r_all':>8}{'r_lo':>8}{'r_hi':>8}{'ms/call':>10}")
    for (d, name), v in rows.items():
        sh = np.mean([x[0] for x in v], 0)
        r_all, r_lo, r_hi = (np.mean([x[i] for x in v]) for i in (1, 2, 3))
        ms = np.median([x[4] for x in v[1:]] or [v[0][4]])          # drop the first call (warm-up)
        lo = f"{r_lo:8.3f}" if np.isfinite(r_lo) else f"{'':>8}"
        hi = f"{r_hi:8.3f}" if np.isfinite(r_hi) else f"{'':>8}"
        print(f"  {d:<5}{name:<20}   ({sh[0]:+5.1f}, {sh[1]:+5.1f}){r_all:8.3f}{lo}{hi}{ms:10.1f}")
    print("\nRead: if a training-free resampler's r_lo matches the learned one, an LF lock (HR follows the"
          "\nLR base at low frequency) needs no trained resizer. ms/call x steps = the cost of a per-step"
          "\nround-trip. Periodic textures (bricks) can make the shift estimate ambiguous.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--images", default="", help="glob; empty = search fix_out/scale_test/seam_out here and one level up")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--no_pixel", action="store_true", help="skip the pixel-space (VAE / taef1) round-trips")
    main(ap.parse_args())
