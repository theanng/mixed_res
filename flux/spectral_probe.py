"""
spectral_probe.py — does the EARLY spectral character of the stage-1 LR trajectory predict the
TRUE octave demand? Ground truth is a SAME-TRAJECTORY uniform-HR continuation (not a separate
full-HR generation), so there is no cross-trajectory divergence confound.

Per prompt:
  1. stage-1 LR denoise (stg1 steps), snapshot x0_hat at each step.
  2. hand the SAME trajectory off to UNIFORM HR (upsample last LR x0_hat + renoise, then denoise
     the whole frame) -> the octave that THIS trajectory actually grows.
  3. octave-residual map on the final HR x0 (down-up latent residual = the octave LR discards).
  4. local spectral-slope map (octave-pyramid) on each early LR x0 snapshot.
  5. Pearson/Spearman of each early slope map vs the ground-truth octave map, printed as a
     saturation curve over early steps -> "how early can we allocate".

    python -m flux.spectral_probe --device cuda:0 --n_prompts 8

Read the curve:
  * |corr| rises and SATURATES by an early step  -> allocation is possible that early (green).
  * |corr| only rises at the last stage-1 step   -> allocate late (progressive promotion needed).
  * |corr| stays ~0                              -> early spectral character does NOT predict late
                                                    demand; a semantic predictor is the real problem.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .run_seam import build_pipe
from .pipeline_flux_base import calculate_shift, retrieve_timesteps
from .seam_prompts import flat


# ---------------- spectral maps (unpacked latent domain [b, C, H, W]) ----------------

def _pool(x, grid):
    return F.adaptive_avg_pool2d(x, grid)[:, 0]                       # [b,1,H,W] -> [b,grid,grid]


def octave_residual_map(lat, grid):
    """Down-up residual energy per cell = the top octave LR would discard (SSS, latent domain)."""
    b, c, H, W = lat.shape
    lo = F.interpolate(lat, scale_factor=0.5, mode="bilinear", align_corners=False)
    up = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
    return _pool(((lat - up) ** 2).mean(1, keepdim=True), grid)


def slope_map(lat, grid, n_oct=3):
    """Local spectral slope via an octave pyramid: energy per octave (finest -> coarsest) fit to a
    log-energy-vs-octave-index line, per grid cell. Steeper (more negative) = energy dies fast =
    smooth; shallower = detail-rich. Robust at coarse grids (no per-patch FFT)."""
    cur = lat
    bands = []
    for _ in range(n_oct):
        lo = F.interpolate(cur, scale_factor=0.5, mode="bilinear", align_corners=False)
        up = F.interpolate(lo, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        bands.append(_pool(((cur - up) ** 2).mean(1, keepdim=True), grid))   # [b,grid,grid]
        cur = lo
    E = torch.stack(bands, 0).clamp(min=1e-12).log()                  # [n_oct, b, grid, grid]
    idx = torch.arange(n_oct, device=lat.device).float()
    idx = idx - idx.mean()
    slope = (E * idx[:, None, None, None]).sum(0) / (idx ** 2).sum()  # [b,grid,grid]
    return -slope                                                    # detail-rich -> positive


def _pearson(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8)


def _spearman(a, b):
    ra = a.reshape(-1).float().argsort().argsort().float()
    rb = b.reshape(-1).float().argsort().argsort().float()
    return _pearson(ra, rb)


def _precision_at_k(pred, truth, k):
    """Top-k overlap: of the k cells the predictor would promote, how many are truly in the top-k
    demand. This is the allocation-relevant metric (unlike Spearman over the whole ranking)."""
    pi = set(pred.reshape(-1).topk(k).indices.tolist())
    ti = set(truth.reshape(-1).topk(k).indices.tolist())
    return len(pi & ti) / max(k, 1)


# ---------------- generation: full LR trajectory + same-trajectory uniform HR ----------------

@torch.no_grad()
def run_one(pipe, args, prompt):
    device = args.device
    bsz = 1
    pipe._h, pipe._w = args.height, args.width
    pipe._c_lt = pipe.transformer.config.in_channels // 4
    gen = torch.Generator("cpu").manual_seed(args.seed)
    grid = args.grid
    budget_k = max(1, int(args.hr_ratio * grid * grid))
    H, W, vsf = args.height, args.width, pipe.vae_scale_factor

    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_embeds=None, pooled_prompt_embeds=None,
        device=device, num_images_per_prompt=1, max_sequence_length=512, lora_scale=None)

    n_ch = pipe.transformer.config.in_channels // 4
    lr_noise, lr_ids = pipe.prepare_latents(bsz, n_ch, H, W, prompt_embeds.dtype, device, gen, None)
    hr_noise, hr_ids = pipe.prepare_latents(bsz, n_ch, H * 2, W * 2, prompt_embeds.dtype, device, gen, None)

    N = args.stg1 + args.stg2
    sigmas = np.linspace(1.0, 1.0 / N, N)
    mu = calculate_shift(hr_noise.shape[1],
                         pipe.scheduler.config.get("base_image_seq_len", 256),
                         pipe.scheduler.config.get("max_image_seq_len", 4096),
                         pipe.scheduler.config.get("base_shift", 0.5),
                         pipe.scheduler.config.get("max_shift", 1.15))
    timesteps_lr, _ = retrieve_timesteps(pipe.scheduler, N, device, sigmas=sigmas, mu=mu)
    timesteps_hr_full, _ = retrieve_timesteps(pipe.scheduler_hr, N, device, sigmas=sigmas)
    stg2_start = N - args.stg2
    timesteps_stg2_hr = timesteps_hr_full[stg2_start:]

    g_hi = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32).expand(bsz)
    g_lo = torch.full([1], 1.0, device=device, dtype=torch.float32).expand(bsz)

    def tf(latents, t, ids, g):
        ts = t.expand(bsz).to(latents.dtype) / 1000
        return pipe.transformer(hidden_states=latents, timestep=ts, guidance=g,
                                pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids, img_ids=ids, joint_attention_kwargs=None,
                                return_dict=False)[0]

    # ---- run LR for the FULL trajectory, capture x0 every step (full saturation curve) ----
    pipe.scheduler.set_begin_index(0)
    latents = lr_noise.clone()
    lr_x0_snaps = []
    cfg_latent, cfg_t = None, None
    for i, t in enumerate(timesteps_lr):
        if i == args.cfg_step:
            cfg_latent, cfg_t = latents.clone(), t
        v = tf(latents, t, lr_ids, g_hi)
        latents, x0 = pipe.scheduler.step(v, t, latents, return_dict=False)
        lr_x0_snaps.append(x0.detach().clone())

    # ---- demand ground truth: take the CLEAN LR to uniform HR (same trajectory) ----
    hr_x0_up, _ = pipe._resize_x0_to_x0(lr_x0_snaps[-1], 2.0, mode="bi", ori_scale=1)
    hr_x0_up = hr_x0_up.to(latents.dtype)
    latents_hr = pipe._add_noise_at_t(hr_x0_up, hr_noise, timesteps_stg2_hr[0])
    pipe.scheduler_hr.set_begin_index(stg2_start)
    hr_x0 = hr_x0_up
    for t in timesteps_stg2_hr:
        v = tf(latents_hr, t, hr_ids, g_hi)
        latents_hr, hr_x0 = pipe.scheduler_hr.step(v, t, latents_hr, return_dict=False)
    hr_lat = pipe._unpack_latents(hr_x0, H * 2, W * 2, vsf).float()
    demand = octave_residual_map(hr_lat, grid)

    # ---- energy predictor: precision@k per LR step (last step = the LR ceiling) ----
    prec, rho = [], []
    for x0 in lr_x0_snaps:
        pred = octave_residual_map(pipe._unpack_latents(x0, H, W, vsf).float(), grid)
        prec.append(_precision_at_k(pred, demand, budget_k))
        rho.append(abs(_spearman(pred, demand).item()))

    out = {"prec": prec, "rho": rho, "ceiling": prec[-1], "budget_k": budget_k}

    # ---- baseline 1: DeepGaze saliency (external, object-aligned) ----
    try:
        lr_img = pipe._vae1_decode(lr_x0_snaps[-1], hw_scale=1, need_unpack=True)
        sal, _ = pipe._get_saliency(lr_img)
        out["gaze"] = _precision_at_k(_pool(sal.float(), grid), demand, budget_k)
    except Exception as e:
        print("  [warn] gaze baseline failed:", e); out["gaze"] = float("nan")

    # ---- baseline 2: CFG-importance |v(g)-v(1)| at an early step (semantic signal) ----
    try:
        d = (tf(cfg_latent, cfg_t, lr_ids, g_hi) - tf(cfg_latent, cfg_t, lr_ids, g_lo)).abs()
        cfg_map = _pool(pipe._unpack_latents(d.float(), H, W, vsf).abs().mean(1, keepdim=True), grid)
        out["cfg"] = _precision_at_k(cfg_map, demand, budget_k)
    except Exception as e:
        print("  [warn] cfg baseline failed:", e); out["cfg"] = float("nan")

    return out


def main(args):
    pipe = build_pipe(args)
    items = list(flat())
    if args.n_prompts > 0:
        items = items[:args.n_prompts]
    rows = []
    for i, (cat, prompt) in enumerate(items):
        o = run_one(pipe, args, prompt)
        rows.append((cat, o))
        print(f"[{i:02d}] {cat[:16]:<16} prec@k/step: " + " ".join(f"{c:.2f}" for c in o["prec"]) +
              f"  | ceil {o['ceiling']:.2f}  gaze {o['gaze']:.2f}  cfg {o['cfg']:.2f}")

    P = np.array([o["prec"] for _, o in rows])
    rand = rows[0][1]["budget_k"] / (args.grid * args.grid)   # precision@k of a random guess
    print("\n=== precision@k (k = budget) vs LR step — energy predictor ===")
    print("LR step:          " + " ".join(f"{k + 1:>5d}" for k in range(P.shape[1])))
    print("energy prec@k:    " + " ".join(f"{v:.2f}" for v in P.mean(0)))
    print(f"\nreference lines:  random={rand:.2f}   "
          f"LR-ceiling(energy@final)={np.mean([o['ceiling'] for _, o in rows]):.2f}   "
          f"DeepGaze={np.nanmean([o['gaze'] for _, o in rows]):.2f}   "
          f"CFG-importance={np.nanmean([o['cfg'] for _, o in rows]):.2f}")
    # per-category means (texture should be ~random -> 'don't split'; objects should be high)
    print("\nper-category ceiling prec@k (energy@final vs demand):")
    cats = {}
    for c, o in rows:
        cats.setdefault(c, []).append(o["ceiling"])
    for c in sorted(cats):
        print(f"  {c[:24]:<24} {np.mean(cats[c]):.2f}  (n={len(cats[c])})")
    print("\nRead: compare energy prec@k to random (floor), DeepGaze/CFG (baselines to beat), and")
    print("the LR-ceiling (best any LR-domain predictor can do). If energy >> gaze, the bandwidth")
    print("thesis holds; if prec@k ~ ceiling early, the limit is informational (decide later/semantic).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--stg1", type=int, default=9, help="LR runs stg1+stg2 steps total (full curve)")
    ap.add_argument("--stg2", type=int, default=9, help="HR refine steps for the demand ground truth")
    ap.add_argument("--lr_shift", type=float, default=1.8776105792643432)
    ap.add_argument("--hr_shift", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=16, help="allocation grid (cells per side)")
    ap.add_argument("--hr_ratio", type=float, default=0.6, help="token budget = hr_ratio * grid^2")
    ap.add_argument("--cfg_step", type=int, default=3, help="LR step index for the CFG-importance baseline")
    ap.add_argument("--n_prompts", type=int, default=-1, help="-1 = all prompts")
    ap.add_argument("--mask_type", choices=["base", "w_seg"], default="base", help="base skips the seg model")
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--out_dir", default="./seam_out")
    main(ap.parse_args())
