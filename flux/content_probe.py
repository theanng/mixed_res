"""
content_probe.py — the professor's ACTUAL hypothesis test: same image content, LR- vs HR-token
representation of one patch, does the information arriving at neighbouring queries differ MORE than
the content change alone would explain?

Unlike channel_probe (holds content fixed, varies Delta = kernel geometry only), this holds the
representation-vs-content trade-off explicit with THREE arms (branch-at-t, from an identical
step-t state; implemented in FluxPipeline._run_content_branch):

  A  : patch LR, original content.
  B  : patch HR  (resizer content)  -> changes representation AND content.
  A' : patch LR, content perturbed by a MATCHED magnitude -> changes content ONLY, same repn.

  The representation effect is  out_chg(A->B) - out_chg(A->A')  at matched content change.
  A' is the falsifiability control: every attention output moves when content moves (A->A' shows
  how much); if B moves MORE per unit content, that excess is the cross-scale representation
  effect -- the professor's hypothesis, measured rather than asserted.

Measurements (per rds>0.085 head; both are cosine distance so comparable):
  - out_chg   = 1 - cos(attn output A vs B / A vs A')   for a NEAR ring of neighbour queries.
  - content   = 1 - cos(patch B-content vs A-content)   (and A'-content vs A-content), over patch.
  - drift     = 1 - cos(neighbour query A vs B)         CONTROL: ~0 at block 0 by construction.
  - out_far   = out_chg for a FAR ring                  NEGATIVE CONTROL: ~0 at block 0.

Reported PER BLOCK, not averaged: block 0 is the clean headline (neighbour queries are provably
identical between arms -> drift 0, pure direct effect). Blocks >=1 mix in propagation (queries have
been updated by block-0 attention) -- the drift curve shows where that contamination begins.

Branches: HRq (LR-key lift: 1 lifted key vs 4 HR keys -- the embedding under test) and LRq (needs
grouping OFF, forced here, so LR queries see the ungrouped 4 HR keys). Geometry: seam patches
(both branches) and/or interior patches (isolated; LRq only, closer to the beard/eye case).

    python -m flux.content_probe --device cuda:0 --geometry both --n_patches 4 --seeds 0,1
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from . import transformer_flux as TF
from .run_seam import build_pipe, run_one
from .comm_prompts import PROMPTS
from .channel_probe import compute_rds, RDS_THRESH
from .mask_patterns import make_mask


def _cdist(x, y):
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean().item()


def _patch_cos_dist(a, b, pm):
    m = pm[..., 0] > 0.5
    return (1.0 - F.cosine_similarity(a[m], b[m], dim=-1)).mean().item()


def _save_attmap(res, branch, patch, path, ht2, wt2):
    """Pick ONE neighbour (the query nearest the patch) and plot ITS attention over the image, so you
    can watch how that single token looks at the promoted region: panels A, B, and (B-A). The
    neighbour is marked with an x; the promoted region is outlined."""
    A = next((r for r in res["A"] if r.get("branch") == branch and "att1" in r), None)
    B = next((r for r in res["B"] if r.get("branch") == branch and "att1" in r), None)
    if A is None or B is None:
        return False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return False

    def grid(rec):
        kc = rec["kcoords"].numpy(); att = rec["att1"].numpy()
        g = np.zeros((ht2, wt2)); c = np.zeros((ht2, wt2))
        yy = np.clip(np.round(kc[:, 0]).astype(int), 0, ht2 - 1)
        xx = np.clip(np.round(kc[:, 1]).astype(int), 0, wt2 - 1)
        np.add.at(g, (yy, xx), att); np.add.at(c, (yy, xx), 1.0)
        return g / np.maximum(c, 1.0)

    gA, gB = grid(A), grid(B)
    qyA, qxA = A["q1coord"].numpy(); qyB, qxB = B["q1coord"].numpy()
    cyh, cxh, pb = patch["cyh"], patch["cxh"], patch["pbox"]

    def box(a, ec):
        a.add_patch(Rectangle((cxh - pb - .5, cyh - pb - .5), 2 * pb + 1, 2 * pb + 1,
                              fill=False, ec=ec, lw=1.4))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7))
    vmax = max(gA.max(), gB.max(), 1e-9)
    for a, g, qy, qx, ttl in [(ax[0], gA, qyA, qxA, f"{branch} A: patch=LR"),
                              (ax[1], gB, qyB, qxB, f"{branch} B: patch=HR")]:
        im = a.imshow(g, cmap="viridis", vmin=0, vmax=vmax); a.set_title(ttl, fontsize=9)
        box(a, "white"); a.plot(qx, qy, "x", color="red", ms=8, mew=2)     # the neighbour
        plt.colorbar(im, ax=a, fraction=0.046)
    d = gB - gA; m = np.abs(d).max() or 1e-9
    im = ax[2].imshow(d, cmap="bwr", vmin=-m, vmax=m); ax[2].set_title("B - A (change)", fontsize=9)
    box(ax[2], "black"); ax[2].plot(qxA, qyA, "x", color="black", ms=8, mew=2)
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)
    return True


def _2d(mask_bool):
    return mask_bool.reshape(mask_bool.shape[-2], mask_bool.shape[-1]).bool()


def _seam_lr_cells(mask_bool):
    """LR cells not core-HR but 4-adjacent to core (the seam): have BOTH HR and LR neighbours."""
    mb = _2d(mask_bool)
    dil = F.max_pool2d(mb.float()[None, None], 3, 1, 1)[0, 0] > 0.5
    return torch.nonzero(dil & (~mb))


def _interior_lr_cells(mask_bool, r=4):
    """LR cells far (>r) from any core-HR: isolated. Promoting them makes a lone HR patch."""
    mb = _2d(mask_bool)
    near = F.max_pool2d(mb.float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0.5
    return torch.nonzero((~mb) & (~near))


def branch_metrics(recs, head_keep, branch):
    """Per-block A/B/A' comparison for one branch. head_keep: per-block bool[H] (or None = all)."""
    idx = {nm: {r["block"]: r for r in recs[nm] if r.get("branch") == branch and not r.get("empty")}
           for nm in ("A", "B", "Ap")}
    out = []
    for blk in sorted(set(idx["A"]) & set(idx["B"]) & set(idx["Ap"])):
        a, b, ap = idx["A"][blk], idx["B"][blk], idx["Ap"][blk]
        keep = head_keep[blk] if (head_keep is not None and blk < len(head_keep)) else None
        if keep is not None and not keep.any():
            continue
        def s(t):
            return t[keep] if keep is not None else t
        rec = dict(block=blk,
                   mass_A=s(a["mass"]).mean().item(), mass_B=s(b["mass"]).mean().item(),
                   mass_Ap=s(ap["mass"]).mean().item(),
                   msgAB=_cdist(s(a["msg"]), s(b["msg"])), msgAAp=_cdist(s(a["msg"]), s(ap["msg"])),
                   outAB=_cdist(s(a["out"]), s(b["out"])), outAAp=_cdist(s(a["out"]), s(ap["out"])),
                   driftAB=_cdist(s(a["q"]), s(b["q"])), driftAAp=_cdist(s(a["q"]), s(ap["q"])))
        if "out_far" in a and "out_far" in b and "out_far" in ap:
            rec["outAB_far"] = _cdist(s(a["out_far"]), s(b["out_far"]))
        out.append(rec)
    return out


def _content_changes(res):
    up, rep, reppert, pm = (res["lt_up_x0"], res["lt_up_x0_rep"],
                            res["lt_up_x0_reppert"], res["pm_hr_tok"])
    return _patch_cos_dist(up, rep, pm), _patch_cos_dist(reppert, rep, pm)   # cc_B, cc_Ap


def main(args):
    pipe = build_pipe(args)
    pipe.transformer.config.lr_rope_downsample_hr_tokens = False   # grouping OFF -> clean coords
    pipe.transformer.lr_group_mode = "off"
    pipe.transformer.hr_lr_expand = False
    pipe.transformer.hr_lr_expand_masscal = False
    pipe.invert_hr_init = False
    pipe.transformer.center_lr = args.center_lr
    pipe.transformer.attn_capture = False
    pipe.custom_mask_pattern = args.mask_pattern
    pipe.custom_mask_hr_ratio = args.mask_hr_ratio
    pipe.custom_mask_strip_w = None
    pipe._content_step = args.step
    pipe._content_ring = float(args.ring)
    pipe._content_noise_match = not args.no_noise_match     # A->B differs only in representation
    if args.map:
        TF._CONTENT["map"] = True; TF._CONTENT["map_block"] = 0

    rds_per_block = [compute_rds(b.attn).cpu() for b in pipe.transformer.transformer_blocks]
    allr = torch.stack(rds_per_block)                                       # [L,H] on cpu
    Hn = allr.shape[1]
    qs = torch.quantile(allr.flatten(), torch.tensor([0.5, 0.9, 0.99]))
    frac = (allr > args.rds_thresh).float().mean().item()
    print(f"rds: mean {allr.mean():.3f}  median {qs[0]:.3f}  p90 {qs[1]:.3f}  max {allr.max():.3f}  "
          f"frac>{args.rds_thresh} = {frac:.2f}")
    if args.head_sel == "all":
        head_keep = None
        print("head selection: ALL heads (FLUX rds is weak vs Wan's threshold; content is RoPE-"
              "independent anyway). Use --head_sel rds/topfrac to restrict.")
    elif args.head_sel == "rds":
        head_keep = [r > args.rds_thresh for r in rds_per_block]
        if frac == 0:
            print("  WARNING: no head exceeds the threshold -> 'rds' yields no data; using --head_sel all is advised.")
    else:  # topfrac: the top args.top_frac heads by rds, per block (relative RoPE-dominance)
        kk = max(1, int(round(args.top_frac * Hn)))
        head_keep = [r >= torch.topk(r, kk).values.min() for r in rds_per_block]
        print(f"head selection: top {kk}/{Hn} heads by rds per block")

    mask = make_mask(args.mask_pattern, args.ht, args.wt, args.mask_hr_ratio, device="cpu")
    if args.patch_yx:                                  # explicit centre (e.g. "16,16" = middle)
        cy, cx = (int(v) for v in args.patch_yx.split(","))
        cells = [("custom", (cy, cx))]
        print(f"placing ONE patch at cell ({cy},{cx}); base mask '{args.mask_pattern}'.")
    else:
        cells = []
        if args.geometry in ("seam", "both"):
            cells += [("seam", tuple(c)) for c in _seam_lr_cells(mask).tolist()]
        if args.geometry in ("interior", "both"):
            cells += [("interior", tuple(c)) for c in _interior_lr_cells(mask).tolist()]
        if not cells:
            raise RuntimeError(f"mask '{args.mask_pattern}' yields no {args.geometry} cells.")
        take = min(args.n_patches, len(cells))
        cells = [cells[i] for i in np.linspace(0, len(cells) - 1, take).astype(int)]
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = [PROMPTS[args.category][args.index]] if args.index >= 0 else PROMPTS[args.category]
    print(f"mask '{args.mask_pattern}' {args.ht}x{args.wt}: probing {len(cells)} cells x "
          f"{len(prompts)} prompts x {len(seeds)} seeds.")

    import os
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    rows = {"HRq": [], "LRq": []}
    curves = {"HRq": {}, "LRq": {}}                     # block -> [(msgAB, drift)...]
    saved = set()
    for seed in seeds:
        args.seed = seed
        for pi, prompt in enumerate(prompts):
            for (geom, (cy, cx)) in cells:
                pipe._content_target = (int(cy), int(cx), int(args.half))
                pipe._content_result = None
                img = run_one(pipe, args, prompt, 0)          # branch fires inside at step t
                if args.out_dir and (seed, pi) not in saved:  # image is unaffected by the branch
                    img.save(os.path.join(args.out_dir, f"s{seed}_p{pi}.png")); saved.add((seed, pi))
                res = getattr(pipe, "_content_result", None)
                if res is None:
                    continue
                if args.map and args.out_dir:
                    for br in ("HRq", "LRq"):
                        p = os.path.join(args.out_dir, f"attmap_{br}_s{seed}_p{pi}_{geom}_{cy}_{cx}.png")
                        _save_attmap(res, br, res["patch"], p, 2 * args.ht, 2 * args.wt)
                cc_B, cc_Ap = _content_changes(res)
                for br in ("HRq", "LRq"):
                    bm = branch_metrics(res, head_keep, br)
                    b0 = next((r for r in bm if r["block"] == 0), None)
                    if b0 is None:
                        continue
                    rows[br].append(dict(geom=geom, cc_B=cc_B, cc_Ap=cc_Ap, **b0))
                    for r in bm:
                        curves[br].setdefault(r["block"], []).append((r["msgAB"], r["driftAB"]))

    for br in ("HRq", "LRq"):
        R = rows[br]
        if not R:
            print(f"\n{br}: no data (empty rings for this geometry)."); continue
        print(f"\n=== {br}: block-0 (clean; mean over {len(R)} patch-runs) ===")
        for geom in sorted(set(r["geom"] for r in R)):
            g = [r for r in R if r["geom"] == geom]
            if not g:
                continue
            mA = np.mean([r["mass_A"] for r in g]); mB = np.mean([r["mass_B"] for r in g])
            mAp = np.mean([r["mass_Ap"] for r in g])
            msgAB = np.mean([r["msgAB"] for r in g]); msgAAp = np.mean([r["msgAAp"] for r in g])
            ccB = np.mean([r["cc_B"] for r in g]); ccAp = np.mean([r["cc_Ap"] for r in g])
            outAB = np.mean([r["outAB"] for r in g]); drift = np.mean([r["driftAB"] for r in g])
            far = np.mean([r.get("outAB_far", np.nan) for r in g])
            print(f"  [{geom:8s} n={len(g):3d}]")
            print(f"     mass onto patch : A(LR)={mA:.4f}  B(HR)={mB:.4f} (x{mB/max(mA,1e-9):.2f})  A'={mAp:.4f}")
            print(f"     patch MESSAGE chg: A->B={msgAB:.3f} (content {ccB:.3f})   "
                  f"A->A'={msgAAp:.3f} (content {ccAp:.3f})   REPN={msgAB - msgAAp:+.3f}")
            print(f"     full-output chg  : A->B={outAB:.4f} (diluted)   controls: drift={drift:.3f} far={far:.3f}")
        blks = sorted(curves[br])
        line = "  ".join(f"b{k}:{np.mean([m for m, _ in curves[br][k]]):.2f}/"
                         f"{np.mean([d for _, d in curves[br][k]]):.3f}" for k in blks[:6])
        print(f"  propagation (block: msgAB/drift): {line}")
    print("\nRead: MESSAGE is the undiluted signal (info the neighbour reads FROM the patch).")
    print("REPN = msg(A->B) - msg(A->A') > 0 => representation distorts info beyond content (SUPPORTED).")
    print("mass B/A ratio = how much MORE the HR patch is attended than the lifted-LR key.")


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
    ap.add_argument("--head_sel", choices=["all", "rds", "topfrac"], default="all",
                    help="which heads to average (all=default; rds=>threshold; topfrac=top by rds)")
    ap.add_argument("--rds_thresh", type=float, default=RDS_THRESH)
    ap.add_argument("--top_frac", type=float, default=0.25, help="head_sel=topfrac: fraction kept")
    ap.add_argument("--geometry", choices=["seam", "interior", "both"], default="interior",
                    help="interior=isolated region (clean for --half>0); seam=boundary (use --half 0)")
    ap.add_argument("--mask_pattern", choices=["contiguous", "bisecting", "strips", "corner"],
                    default="corner", help="corner=tiny HR core in corner -> middle stays LR for a centred patch")
    ap.add_argument("--patch_yx", default="", help='place ONE patch at this LR cell, e.g. "16,16" (middle)')
    ap.add_argument("--mask_hr_ratio", type=float, default=0.05,
                    help="area of the base HR core (kept small so it's out of the way of the patch)")
    ap.add_argument("--ht", type=int, default=32, help="LR latent grid height (512 -> 32)")
    ap.add_argument("--wt", type=int, default=32)
    ap.add_argument("--half", type=int, default=3,
                    help="patch half-size in LR cells (0=1 cell; 3=7x7 REGION -> non-negligible mass)")
    ap.add_argument("--ring", type=float, default=4.0, help="neighbour ring width (HR-grid units)")
    ap.add_argument("--step", type=int, default=4, help="stage-2 step to branch at (past promotion)")
    ap.add_argument("--center_lr", action="store_true")
    ap.add_argument("--no_noise_match", action="store_true",
                    help="disable noise-matching (then A->B is confounded by a different noise seed)")
    ap.add_argument("--map", action="store_true", default=True, help="save neighbour attention maps")
    ap.add_argument("--n_patches", type=int, default=4)
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--category", default="face_bisected")
    ap.add_argument("--index", type=int, default=-1, help="-1 = all prompts in the category")
    ap.add_argument("--out_dir", default="content_out", help="save images + attention maps")
    main(ap.parse_args())
