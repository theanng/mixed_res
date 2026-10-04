"""
channel_probe.py — CRPA's content-free phase kernel + rds head selection.

  mode=kernel  reproduces CRPA Fig. 3: the single-stride phase kernel
                 kappa(Delta) = E_i[ <q_hat_i, R(Delta) k_hat_i> ]  vs offset Delta, over
                 same-position q,k pairs. Sharp peak at 0, periodic. All-heads and rds>0.085.

  mode=sens    kernel-GEOMETRY sensitivity to a half-cell index error:
                 mean_d |kappa(d + centroid) - kappa(d)|. This quantifies the top-left-vs-center
                 (--center_lr) question: "how much does the kernel move under a 0.5-cell offset."
                 NOTE: this is a property of the kernel shape, NOT a cross-scale attention
                 measurement. It cannot test whether the cross-scale EMBEDDING is wrong, because
                 it holds content fixed and only shifts Delta. For the actual embedding/content
                 hypothesis ("same image content, LR-token vs HR-token representation, does the
                 information arriving at neighbouring queries differ?"), see flux/content_probe.py.

  rds(h) = (2/d) sum_i cos(W_2i, W_2i+1)  over the query-proj rotary pairs (weights only,
           no forward). rds>0.085 = RoPE-dominant (paper's threshold, from [15]).

    python -m flux.channel_probe --device cuda:0 --axis w --mode kernel
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from . import transformer_flux as TF
from .transformer_flux import apply_rotary_emb
from .run_seam import build_pipe, run_one
from .comm_prompts import PROMPTS

RDS_THRESH = 0.085


# ----------------------------------------------------------------------------- rds (weights only)
def compute_rds(attn):
    """rds per head from the query-projection weights: mean cos of each rotary pair's two rows."""
    W = attn.to_q.weight.data.float()                       # [heads*head_dim, d_model]
    H = attn.heads
    Wh = W.view(H, -1, W.shape[1])                          # [H, head_dim, d_model]
    d = Wh.shape[1]
    a = Wh[:, 0:d:2, :]                                     # rows 2i   [H, d/2, d_model]
    b = Wh[:, 1:d:2, :]                                     # rows 2i+1
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-9)   # [H, d/2]
    return cos.mean(-1)                                     # [H]


def rds_report(pipe):
    blocks = list(pipe.transformer.transformer_blocks)
    rds_per_block = [compute_rds(b.attn) for b in blocks]   # each [H]
    allr = torch.stack(rds_per_block)                       # [L, H]
    frac = (allr > RDS_THRESH).float().mean().item()
    print(f"=== rds over {allr.shape[0]} double-blocks x {allr.shape[1]} heads ===")
    print(f"  mean rds = {allr.mean():.3f}   max = {allr.max():.3f}   "
          f"frac RoPE-dominant (>{RDS_THRESH}) = {frac:.2f}")
    return rds_per_block


# --------------------------------------------------------------------- shared kappa machinery
@torch.no_grad()
def _sample_pairs(q, k, n_pairs, head_keep=None):
    """Co-occurring (same-position) q,k pairs, L2-normalized. q,k: [seq,H,D] -> qs,ks: [P,H,D].

    NOTE: q and k are paired at the SAME token index. This is essential: for INDEPENDENT random
    q,k, E[<q_hat, R(Delta) k_hat>] = 0 for every Delta (a rotated random vector is uncorrelated),
    so the kernel would be flat. The peak/periodicity in CRPA's kappa(Delta) comes from the head's
    LEARNED, consistent phase relation between its q- and k-projections across positions; content
    only modulates amplitude. Averaging same-position pairs over positions isolates that phase
    prior => content-averaged (the phase structure is a property of W_q,W_k, not the content)."""
    seq = q.shape[0]
    if head_keep is not None:
        q, k = q[:, head_keep], k[:, head_keep]
    qn = F.normalize(q, dim=-1)
    kn = F.normalize(k, dim=-1)
    idx = torch.randint(0, seq, (n_pairs,))
    return qn[idx], kn[idx]


@torch.no_grad()
def _kappa_at(qs, ks, pos_embed, axis, delta):
    """Apply a SINGLE RoPE offset `delta` (float, along `axis`) to every key, dot with query.
    Returns per-pair-per-head cosine [P,H] (not yet averaged)."""
    coords = {"t": torch.zeros(1), "h": torch.zeros(1), "w": torch.zeros(1)}
    coords[axis] = torch.tensor([float(delta)])
    cos, sin = pos_embed.get_rotary_for_coords(coords["t"], coords["h"], coords["w"])  # [1,D]
    cos, sin = cos.to(ks.dtype), sin.to(ks.dtype)
    kr = apply_rotary_emb(ks.unsqueeze(0), (cos, sin)).squeeze(0)   # [P,H,D]
    return (qs * kr).sum(-1)                                        # [P,H]


@torch.no_grad()
def kappa_curve(q, k, pos_embed, axis, deltas, head_keep=None, n_pairs=4000):
    """The single-stride phase kernel: kappa(Delta) averaged over content pairs & heads."""
    qs, ks = _sample_pairs(q, k, n_pairs, head_keep)
    return np.array([_kappa_at(qs, ks, pos_embed, axis, d).mean().item() for d in deltas])


@torch.no_grad()
def centroid_sensitivity(q, k, pos_embed, axis, phys, centroid=0.5, head_keep=None, n_pairs=4000):
    """How much the kernel MOVES under a half-cell index error: kappa(d) and kappa(d+centroid).
    This is kernel geometry only (it does NOT vary content), so it quantifies the top-left-vs-center
    (--center_lr) question but is NOT a cross-scale attention/embedding measurement."""
    qs, ks = _sample_pairs(q, k, n_pairs, head_keep)
    k0 = np.array([_kappa_at(qs, ks, pos_embed, axis, d).mean().item() for d in phys])
    koff = np.array([_kappa_at(qs, ks, pos_embed, axis, d + centroid).mean().item() for d in phys])
    return k0, koff


def _save_plot(deltas, curves, path, xlabel="Delta (offset)", title="kappa(Delta): phase kernel"):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("  (no matplotlib:", e, ")"); return
    plt.figure(figsize=(6.4, 3.8))
    for name, y in curves.items():
        plt.plot(deltas, y, label=name, marker="." if len(deltas) < 40 else None)
    plt.axvline(0, color="k", lw=0.5); plt.xlabel(xlabel); plt.ylabel("kappa")
    plt.legend(fontsize=8); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); print("  saved ->", path)


def main(args):
    pipe = build_pipe(args)
    rds_per_block = rds_report(pipe)

    # capture real content q/k (post-norm, PRE-RoPE) from the first blocks of a short forward
    TF._KAPPA["on"] = True; TF._KAPPA["q"].clear(); TF._KAPPA["k"].clear()
    TF._KAPPA["max"] = args.n_layers
    run_one(pipe, args, PROMPTS[args.category][args.index], 0)
    TF._KAPPA["on"] = False
    caps_q, caps_k = TF._KAPPA["q"], TF._KAPPA["k"]
    print(f"\ncaptured q/k from {len(caps_q)} blocks (each {tuple(caps_q[0].shape)})")

    pe = pipe.transformer.pos_embed

    def _rds_keep(i):
        return (rds_per_block[i] > RDS_THRESH) if i < len(rds_per_block) else None

    # ------------------------------------------------------------------ mode=kernel: CRPA Fig. 3
    if args.mode in ("kernel", "both"):
        deltas = list(range(0, args.max_delta))
        all_c, rope_c = [], []
        for i, (q, k) in enumerate(zip(caps_q, caps_k)):
            q, k = q[0], k[0]
            keep = _rds_keep(i)
            all_c.append(kappa_curve(q, k, pe, args.axis, deltas, n_pairs=args.n_pairs))
            if keep is not None and keep.any():
                rope_c.append(kappa_curve(q, k, pe, args.axis, deltas, head_keep=keep, n_pairs=args.n_pairs))
        curves = {"all heads": np.mean(all_c, 0)}
        if rope_c:
            curves[f"rds>{RDS_THRESH}"] = np.mean(rope_c, 0)
        print("\n=== kappa(Delta): single-stride phase kernel (CRPA Fig. 3) ===")
        print("Delta:  " + " ".join(f"{d:>5d}" for d in deltas[:12]))
        for name, y in curves.items():
            print(f"{name:<12} " + " ".join(f"{v:>5.2f}" for v in y[:12]))
        print("Expect: sharp peak at 0, drop within 2-3 offsets, periodic; rds heads amplified.")
        _save_plot(deltas, curves, args.plot)

    # -------------------------------- mode=sens: half-cell kernel sensitivity (--center_lr question)
    if args.mode in ("sens", "both"):
        phys = np.round(np.arange(0.0, args.phys_max + 1e-6, 0.5), 3)
        k0s, koffs, n_used = [], [], 0
        for i, (q, k) in enumerate(zip(caps_q, caps_k)):
            keep = _rds_keep(i)
            if keep is None or not keep.any():
                continue
            a, b = centroid_sensitivity(q[0], k[0], pe, args.axis, phys,
                                        centroid=args.centroid, head_keep=keep, n_pairs=args.n_pairs)
            k0s.append(a); koffs.append(b); n_used += 1
        if not k0s:                                   # no rds heads -> all heads
            for i, (q, k) in enumerate(zip(caps_q, caps_k)):
                a, b = centroid_sensitivity(q[0], k[0], pe, args.axis, phys,
                                            centroid=args.centroid, n_pairs=args.n_pairs)
                k0s.append(a); koffs.append(b); n_used += 1
        k0, koff = np.mean(k0s, 0), np.mean(koffs, 0)
        sens = float(np.mean(np.abs(koff - k0)))
        print(f"\n=== half-cell kernel sensitivity (KERNEL GEOMETRY, not a channel test) ===")
        print(f"  mean_d |kappa(d+{args.centroid:g}) - kappa(d)| = {sens:.3f}   "
              f"(axis={args.axis}, {n_used} block(s), rds-heads)")
        print(f"  => a top-left vs centered ({args.centroid:g}-cell) index error moves the kernel by ~{sens:.3f}.")
        print(f"     This bounds what --center_lr can do; it does NOT test the embedding hypothesis.")
        print(f"     For that, run: python -m flux.content_probe   (LR-vs-HR representation of one patch)")
        _save_plot(phys, {"kappa(d)": k0, f"kappa(d+{args.centroid:g})": koff},
                   args.plot.replace(".png", "_sens.png"),
                   xlabel="physical distance d (HR cells)", title="half-cell kernel sensitivity")


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
    # probe
    ap.add_argument("--mode", choices=["kernel", "sens", "both"], default="kernel")
    ap.add_argument("--axis", choices=["t", "h", "w"], default="w")
    ap.add_argument("--max_delta", type=int, default=32, help="offset range for mode=kernel")
    ap.add_argument("--phys_max", type=float, default=24.0, help="max physical distance for mode=sens (HR cells)")
    ap.add_argument("--centroid", type=float, default=0.5, help="LR footprint-centroid offset in HR cells (top-left bug)")
    ap.add_argument("--n_pairs", type=int, default=4000, help="# random content pairs per block (the expectation)")
    ap.add_argument("--n_layers", type=int, default=8, help="# blocks to capture q/k from")
    ap.add_argument("--category", default="face_bisected")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--plot", default="kappa.png")
    main(ap.parse_args())
