"""
attn_probe.py — measure whether HR queries attend to their own kind more than to LR.

Runs one mixed-res generation with attention capture on, then aggregates, per key-TYPE
(text / HR / LR), the mean softmax mass, the per-key mass (count-normalized), and the mean
pre-softmax logit -- for both HR-queries and LR-queries. No token x token dump.

The decisive numbers (HR-query row):
  * per-key HR/LR ratio > 1  -> HR queries PREFER HR keys over LR keys at equal count
                                (the "attends to own kind" bias -- the beard hypothesis).
  * logit gap (HR - LR) > 0  -> it's a pre-softmax MAGNITUDE/distribution effect
                                (HR-HR dot products are simply larger than HR-LR).

    # match the beard case (isolated 4x4 HR patches on a face), E&R off, ALL arm:
    python -m flux.attn_probe --device cuda:0 --category face_bisected --index 1 \
        --mask_pattern scattered --strip_w 2 --group_mode off

    # or the fat-blob status quo for comparison:
    python -m flux.attn_probe --device cuda:0 --category face_bisected --index 1 \
        --mask_pattern contiguous --group_mode off
"""

import argparse

import numpy as np
import torch

from . import transformer_flux as TF
from .run_seam import build_pipe, run_one
from .comm_prompts import PROMPTS


def _summarize(log, branch):
    recs = [r for r in log if r.get("branch") == branch]
    if not recs:
        return None, 0
    out = {}
    for key in ["txt", "HR", "LR"]:
        vals = lambda f: [r[f"{key}_{f}"] for r in recs if f"{key}_{f}" in r]
        if not vals("mass"):
            continue
        out[key] = (float(np.mean(vals("mass"))), float(np.mean(vals("perkey"))),
                    float(np.mean(vals("logit"))), float(np.mean(vals("n"))))
    return out, len(recs)


def _save_plot(hrq, lrq, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable, skipping plot:", e, ")"); return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (title, d) in zip(axes, [("HR queries", hrq), ("LR queries", lrq)]):
        if not d:
            continue
        ks = [k for k in ["txt", "HR", "LR"] if k in d]
        ax.bar(ks, [d[k][1] for k in ks], color=["#999", "#c33", "#39c"])
        ax.set_title(title); ax.set_ylabel("per-key attention")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print("  saved plot ->", path)


def _perkey(md):
    """per-key attention vs distance = mass / count (isolates decay from key layout)."""
    mass, cnt = md
    return mass / cnt.clamp(min=1.0)


def _reach(md, binw, frac=0.5):
    pk = _perkey(md)
    m = pk / pk.sum().clamp(min=1e-9)
    b = int((m.cumsum(0) >= frac).float().argmax().item())
    return (b + 0.5) * binw


def _save_dist_plot(prof, binw, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable:", e, ")"); return
    x = (np.arange(len(next(iter(prof.values())))) + 0.5) * binw
    plt.figure(figsize=(5, 3.4))
    for br, c in [("HRq", "#c33"), ("LRq", "#39c")]:
        if br in prof:
            plt.plot(x, prof[br], label=br, color=c)
    plt.xlabel("physical distance (HR-grid units)"); plt.ylabel("attention mass (norm.)")
    plt.legend(); plt.title("cross-scale physical reach"); plt.tight_layout()
    plt.savefig(path, dpi=120); print("  saved dist plot ->", path)


def _save_spatial(spatial, path):
    """Scatter each captured query's attention over the image keys (HR-grid coords), marking the
    query. HR keys drawn as circles, LR keys as squares, colour = attention weight."""
    if not spatial:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable:", e, ")"); return
    n = len(spatial)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, s in enumerate(spatial):
        ax = axes[i // cols][i % cols]
        kc, w, nhr = s["kc"], s["w"], s["n_hr"]
        # HR keys = first n_hr rows, LR keys = rest (matches capture order)
        ax.scatter(kc[:nhr, 1], kc[:nhr, 0], c=w[:nhr], s=14, marker="o", cmap="magma")
        ax.scatter(kc[nhr:, 1], kc[nhr:, 0], c=w[nhr:], s=22, marker="s", cmap="magma")
        ax.plot(s["q"][1], s["q"][0], "x", color="lime", markersize=12, markeredgewidth=2)
        ax.invert_yaxis(); ax.set_title(f"{s['branch']} q@({s['q'][0]:.0f},{s['q'][1]:.0f})", fontsize=8)
        ax.set_aspect("equal")
    fig.suptitle("attention weight per key (o=HR key, s=LR key, x=query)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); print("  saved spatial map ->", path)


def main(args):
    pipe = build_pipe(args)
    pipe.transformer.config.lr_rope_downsample_hr_tokens = (args.group_mode != "off")
    pipe.transformer.lr_group_mode = args.group_mode
    pipe.transformer.hr_lr_expand = False
    pipe.transformer.hr_lr_expand_masscal = False
    pipe.invert_hr_init = False
    pipe.custom_mask_pattern = None if args.mask_pattern == "saliency" else args.mask_pattern
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.custom_mask_strip_w = args.strip_w if args.strip_w > 0 else None
    pipe.transformer.attn_capture = True

    prompt = PROMPTS[args.category][args.index]
    print("prompt:", prompt, "\nmask:", args.mask_pattern, "strip_w", args.strip_w, "group", args.group_mode)
    TF._ATTN_LOG.clear()
    TF._DIST_LOG.clear(); TF._DIST_CNT["n"] = 0
    TF._SPATIAL.clear()
    run_one(pipe, args, prompt, 0)                      # pad=0 -> E&R off
    log = list(TF._ATTN_LOG)
    print(f"captured {len(log)} branch-records\n")

    summaries = {}
    for branch in ["HRq", "LRq"]:
        d, n = _summarize(log, branch)
        summaries[branch] = d
        if not d:
            print(f"[{branch}] no records\n"); continue
        print(f"=== {branch}  (n={n} records) ===")
        print(f"{'key-type':<9}{'mass':>10}{'per-key':>12}{'logit':>10}{'#keys':>8}")
        for k in ["txt", "HR", "LR"]:
            if k in d:
                m, pk, lo, cnt = d[k]
                print(f"{k:<9}{m:>10.3f}{pk:>12.2e}{lo:>10.3f}{cnt:>8.0f}")
        if "HR" in d and "LR" in d:
            print(f"  -> per-key HR/LR = {d['HR'][1] / max(d['LR'][1], 1e-12):.2f}"
                  f"  (>1 = prefers own kind)   logit gap HR-LR = {d['HR'][2] - d['LR'][2]:+.3f}")
        print()

    _save_plot(summaries.get("HRq"), summaries.get("LRq"), args.plot)

    # --- physical reach (the reach-asymmetry test) ---
    dl = TF._DIST_LOG
    if dl:
        binw = TF._DIST_BIN_W
        prof = {br: (lambda p: (p / max(p.sum().item(), 1e-9)).numpy())(_perkey(dl[br])) for br in dl}
        print("\n=== physical reach (PER-KEY attention vs distance, HR-grid units) ===")
        for br in ["HRq", "LRq"]:
            if br in dl:
                print(f"  {br}: 50%-reach = {_reach(dl[br], binw):.1f}   "
                      f"75%-reach = {_reach(dl[br], binw, 0.75):.1f}")
        if "HRq" in dl and "LRq" in dl:
            r = _reach(dl["LRq"], binw) / max(_reach(dl["HRq"], binw), 1e-9)
            print(f"  -> LR/HR 50%-reach ratio = {r:.2f}   (reach-asymmetry predicts ~2.0)")
        _save_dist_plot(prof, binw, args.dist_plot)

    _save_spatial(list(TF._SPATIAL), args.spatial_plot)

    print("\nRead: per-key HR/LR ~ locality (drops when scattered). The decisive number is the")
    print("LR/HR reach ratio: ~2 => HR queries physically reach half as far (CRPA spends kappa's")
    print("token budget in the fine stride) => isolated HR patch can't reach the coarse context.")


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
    ap.add_argument("--group_mode", choices=["off", "topleft", "mean"], default="off")
    ap.add_argument("--mask_pattern", choices=["saliency", "full", "contiguous", "bisecting", "strips", "scattered"],
                    default="scattered")
    ap.add_argument("--mask_hr_ratio", type=float, default=0.3)
    ap.add_argument("--strip_w", type=int, default=2)
    ap.add_argument("--category", default="face_bisected")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--invert_steps", type=int, default=4)
    ap.add_argument("--boundary_grad", type=int, default=0)
    ap.add_argument("--plot", default="attn_probe.png")
    ap.add_argument("--dist_plot", default="attn_reach.png")
    ap.add_argument("--spatial_plot", default="attn_spatial.png")
    main(ap.parse_args())
