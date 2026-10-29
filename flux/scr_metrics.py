"""
scr_metrics.py -- Phase Q / X5 metrics, offline, on saved run folders (numpy / scipy / PIL only).

Per case folder <root>/<tag>/<category>_<i>/ with mixed.png + mask.npy (and, if the run used
--scr_dump_streams, the raw stream decodes dec_hr.png / dec_lr.png):

  tex_ratio   texture (row) period inside the HR core / outside it. Patches of 128 px (fallback 96, 64)
              fully inside the core on raw dec_hr, and >= 32 px outside it on dec_lr (bicubic x2); periods
              from the row-profile autocorrelation, kept if periodicity >= 0.4; medians. 1.0 = same scale,
              0.4 = the "mini-bricks" of plain CRPA / E&R / C. Without dumps, mixed.png is used past the feather.
  lock8       grid lock at P = 8 px (one HR latent pixel): fraction of the high-pass luminance residual that is a
              pattern repeated every 8 px. Core (raw dec_hr, or mixed.png eroded 32 px) and outside
              (mixed.png >= 32 px outside the core). Off-manifold latents decode as such a pattern (Phase Q Q2).
  aniso       Ex/Ey of luminance gradients per 32-px core cell (vertical streaks >> 1), median over cells.
  bnd_col / bnd_ori / bnd_logE
              boundary descriptor: bands 32-64 px inside vs outside the core edge on mixed.png, per 64-px tile:
              mean Lab colour distance, gradient-orientation histogram L1 (0-1), log gradient-energy ratio.
              Colour / orientation = structural disagreement (P2); logE = detail level (octave seam).
  seam        P2 (structural disagreement), RAW streams only: hard composite (HR projected to the LR band in, LR out); jump across
              the edge / the LR image's own variation on the same pixel pairs. 1 = no seam, > 1 = disagreement.
  eps_*       (relay / e6 runs with --scr_log) whiteness of the HR stream's own noise estimate in the core per
              stage-2 step, from meta.json: band densities b0..b3 (white = 1) and lag-1 autocorrelation (white = 0).

    python -m flux.scr_metrics ./phaseQ_out/main                    # table per run
    python -m flux.scr_metrics ./phaseQ_out/main --csv m.csv -v     # + per-case rows, CSV
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------------------------------------- basics
def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


def luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def rgb2lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# --------------------------------------------------------------------------------------------- grid lock
def grid_lock(y, mask, P, highpass=True, sigma=2.0, erode=0):
    """Fraction of the (high-passed) residual that repeats with period P in both axes inside mask.
    amp = rms of the P x P folded mean / rms of the residual; z = amp * sqrt(mean count per bin) (~1 if unlocked)."""
    m = mask.astype(bool)
    if erode:
        m = ndimage.binary_erosion(m, iterations=erode)
    if m.sum() < 4 * P * P:
        return None
    r = y - ndimage.gaussian_filter(y, sigma) if highpass else y
    yy, xx = np.nonzero(m)
    v = r[yy, xx] - r[yy, xx].mean()
    rms = v.std()
    if rms < 1e-9:
        return None
    idx = (yy % P) * P + (xx % P)
    cnt = np.bincount(idx, minlength=P * P)
    tot = np.bincount(idx, weights=v, minlength=P * P)
    ok = cnt > 0
    t = tot[ok] / cnt[ok]
    amp = float(np.sqrt((t ** 2).mean()) / rms)
    return {"amp": round(amp, 4), "z": round(amp * float(np.sqrt(cnt[ok].mean())), 2), "n": int(m.sum())}


# --------------------------------------------------------------------------------------------- periods
def row_period(p, lo=6, hi=None, min_corr=0.4):
    """Period (px) of the row profile (mean over columns) of patch p: first autocorrelation peak that is
    >= min_corr and >= 0.8 x the best peak (avoids harmonics). Returns (period, corr) or None."""
    prof = p.mean(1) - p.mean()
    n = len(prof)
    if n < 2 * lo or prof.std() < 1e-6:
        return None
    hi = min(hi or n // 2, n - 2)
    ac = np.correlate(prof, prof, "full")[n - 1:]
    ac = ac / ac[0]
    cand = [k for k in range(max(lo, 1), hi) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] >= min_corr]
    if not cand:
        return None
    best = max(ac[k] for k in cand)
    k = next(k for k in cand if ac[k] >= 0.8 * best)
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    den = a - 2 * b + c
    kf = k + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return float(kf), float(b)


def _patch_corners(ok, size, stride):
    """Top-left corners of size x size windows fully inside boolean map ok (integral image)."""
    S = np.pad(np.cumsum(np.cumsum((~ok).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    H, W = ok.shape
    out = []
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            if S[y + size, x + size] - S[y, x + size] - S[y + size, x] + S[y, x] == 0:
                out.append((y, x))
    return out


def tex_ratio(y_in, y_out, mask, gap_in=0, gap_out=32, stride=32, min_corr=0.4):
    """Median texture period inside the core (on y_in) / outside (on y_out, same rows when possible)."""
    m = mask.astype(bool)
    din = ndimage.distance_transform_edt(m)
    dout = ndimage.distance_transform_edt(~m)
    for size in (128, 96, 64):
        ins = _patch_corners(din > gap_in, size, stride)
        if ins:
            break
    if not ins:
        return None
    outs = _patch_corners(dout > gap_out, size, stride)
    rows = (min(y for y, _ in ins) - size, max(y for y, _ in ins) + size)
    same = [(y, x) for y, x in outs if rows[0] <= y <= rows[1]]
    outs = same or outs
    pin = [row_period(y_in[y:y + size, x:x + size], min_corr=min_corr) for y, x in ins]
    pout = [row_period(y_out[y:y + size, x:x + size], min_corr=min_corr) for y, x in outs]
    pin = [p[0] for p in pin if p]
    pout = [p[0] for p in pout if p]
    res = {"size": size, "n_in": len(pin), "n_out": len(pout)}
    if pin and pout:
        res.update(p_in=round(float(np.median(pin)), 2), p_out=round(float(np.median(pout)), 2))
        res["ratio"] = round(res["p_in"] / res["p_out"], 3)
    return res


# --------------------------------------------------------------------------------------------- streaks
def aniso(y, mask, cell=32):
    gx = np.pad((y[:, 1:] - y[:, :-1]) ** 2, ((0, 0), (0, 1)))
    gy = np.pad((y[1:, :] - y[:-1, :]) ** 2, ((0, 1), (0, 0)))
    H, W = y.shape
    r = []
    for cy in range(0, H - cell + 1, cell):
        for cx in range(0, W - cell + 1, cell):
            if mask[cy:cy + cell, cx:cx + cell].all():
                ey = gy[cy:cy + cell, cx:cx + cell].mean()
                if ey > 1e-9:
                    r.append(gx[cy:cy + cell, cx:cx + cell].mean() / ey)
    return {"median": round(float(np.median(r)), 3), "max": round(float(np.max(r)), 2), "n": len(r)} if r else None


# --------------------------------------------------------------------------------------------- boundary
def boundary(rgb, mask, band=(32, 64), tile=64, min_px=200, bins=8):
    m = mask.astype(bool)
    din = ndimage.distance_transform_edt(m)
    dout = ndimage.distance_transform_edt(~m)
    bin_ = (din >= band[0]) & (din <= band[1])
    bout = (dout >= band[0]) & (dout <= band[1])
    lab = rgb2lab(rgb)
    y = luma(rgb)
    gy, gx = np.gradient(y)
    mag = np.hypot(gx, gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)
    col, orid, loge = [], [], []
    H, W = m.shape
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            sl = (slice(max(0, ty - tile // 2), ty + tile + tile // 2), slice(max(0, tx - tile // 2), tx + tile + tile // 2))
            a, b = bin_[sl], bout[sl]
            if a.sum() < min_px or b.sum() < min_px:
                continue
            col.append(float(np.linalg.norm(lab[sl][a].mean(0) - lab[sl][b].mean(0))))
            ha, _ = np.histogram(ori[sl][a], bins, (0, np.pi), weights=mag[sl][a])
            hb, _ = np.histogram(ori[sl][b], bins, (0, np.pi), weights=mag[sl][b])
            if ha.sum() > 0 and hb.sum() > 0:
                orid.append(float(0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum()))
            ea, eb = (mag[sl][a] ** 2).mean(), (mag[sl][b] ** 2).mean()
            if ea > 0 and eb > 0:
                loge.append(float(np.log(ea / eb)))
    if not col:
        return None
    return {"col": round(float(np.median(col)), 3), "ori": round(float(np.median(orid)), 3) if orid else None,
            "logE": round(float(np.median(loge)), 3) if loge else None, "n": len(col)}


# --------------------------------------------------------------------------------------------- seam (P2)
def lr_band(rgb, lr_size=512):
    """Project an image onto the LR stream's band: area-downsample to the LR resolution, bicubic back up
    (the same path dec_lr takes into the final render). Removes the HR octave, so only disagreement remains."""
    H = rgb.shape[0]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    im = im.resize((lr_size, lr_size), Image.BOX).resize((H, H), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


def seam_jump(rgb_hr, rgb_lr, mask, k=3, sigma=1.0):
    """Structural disagreement across the core edge, from the RAW streams (no feather).
    Hard composite = HR projected to the LR band (lr_band) inside, LR (bicubic x2) outside, light Gaussian.
    For pixel pairs (p inside, q outside) k px apart across the edge (4 directions):
        seam = mean |Lab(comp p) - Lab(comp q)| / mean |Lab(lr p) - Lab(lr q)|  (same pairs, LR image alone)
    1 = the edge is no more of a jump than the LR image's own variation there; > 1 = the two owners
    disagree at the boundary (ghost eyes, broken contours, colour steps). The HR octave (sharpness) is
    removed first, so a sharp but aligned HR region scores ~1."""
    m = mask.astype(bool)
    lab_h = np.stack([ndimage.gaussian_filter(c, sigma) for c in np.moveaxis(rgb2lab(lr_band(rgb_hr)), -1, 0)], -1)
    lab_l = np.stack([ndimage.gaussian_filter(c, sigma) for c in np.moveaxis(rgb2lab(rgb_lr), -1, 0)], -1)
    comp = np.where(m[..., None], lab_h, lab_l)
    num, den = [], []
    for dy, dx in ((0, k), (0, -k), (k, 0), (-k, 0)):
        mq = np.roll(m, (-dy, -dx), (0, 1))                       # mq[p] = m[p + d]
        sel = m & ~mq
        if dy > 0: sel[-dy:, :] = False
        if dy < 0: sel[:-dy, :] = False
        if dx > 0: sel[:, -dx:] = False
        if dx < 0: sel[:, :-dx] = False
        ys, xs = np.nonzero(sel)
        if ys.size == 0:
            continue
        num.append(np.linalg.norm(comp[ys, xs] - comp[ys + dy, xs + dx], axis=-1))
        den.append(np.linalg.norm(lab_l[ys, xs] - lab_l[ys + dy, xs + dx], axis=-1))
    if not num:
        return None
    num, den = np.concatenate(num), np.concatenate(den)
    return {"ratio": round(float(num.mean() / max(den.mean(), 1e-6)), 3), "jump": round(float(num.mean()), 3),
            "base": round(float(den.mean()), 3), "n": int(num.size)}


# --------------------------------------------------------------------------------------------- per case
def case_metrics(d):
    mixed = load_rgb(os.path.join(d, "mixed.png"))
    mask = np.load(os.path.join(d, "mask.npy")).astype(bool)
    if mask.shape != mixed.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(mixed.shape[1::-1])) > 127
    has = os.path.exists(os.path.join(d, "dec_hr.png")) and os.path.exists(os.path.join(d, "dec_lr.png"))
    ym = luma(mixed)
    seam = None
    if has:
        rgb_hr = load_rgb(os.path.join(d, "dec_hr.png"))
        rgb_lr = load_rgb(os.path.join(d, "dec_lr.png"), size=mixed.shape[0])
        y_hr, y_lr = luma(rgb_hr), luma(rgb_lr)
        seam = seam_jump(rgb_hr, rgb_lr, mask)
        tr = tex_ratio(y_hr, y_lr, mask, gap_in=0)
        core_y, core_m = y_hr, mask
    else:                                   # no raw streams: use mixed.png past the 28 px feather
        tr = tex_ratio(ym, ym, mask, gap_in=40, gap_out=40)
        core_y, core_m = ym, ndimage.binary_erosion(mask, iterations=32)
    out_m = ndimage.distance_transform_edt(~mask) >= 32
    r = {"streams": has, "tex": tr,
         "lock8_core": grid_lock(core_y, core_m, 8), "lock8_out": grid_lock(ym, out_m, 8),
         "aniso": aniso(core_y, core_m), "bnd": boundary(mixed, mask), "seam": seam}
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):
        scr = json.load(open(meta_p)).get("scr") or {}
        eps = [s.get("eps") for s in scr.get("steps", []) if s.get("eps")]
        if eps:
            r["eps"] = eps
        if "last_write_grid_lock" in scr:
            r["last_write_lock"] = scr["last_write_grid_lock"]
    return r


def _get(r, *keys):
    for k in keys:
        if r is None:
            return None
        r = r.get(k) if isinstance(r, dict) else None
    return r


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt(x, n=2):
    return "   -  " if x is None else f"{x:6.{n}f}"


def main(args):
    rows = []
    for root in args.roots:
        for run in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(run) or (args.runs and not any(s in os.path.basename(run) for s in args.runs)):
                continue
            cases = sorted(c for c in glob.glob(os.path.join(run, "*"))
                           if os.path.exists(os.path.join(c, "mixed.png")) and os.path.exists(os.path.join(c, "mask.npy")))
            for c in cases:
                r = case_metrics(c)
                r.update(run=os.path.basename(run), case=os.path.basename(c), root=root)
                rows.append(r)
                if args.verbose:
                    print(f"  {r['run']:<34} {r['case']:<22} tex {_fmt(_get(r, 'tex', 'ratio'))}  "
                          f"lock8 {_fmt(_get(r, 'lock8_core', 'amp'), 3)}/{_fmt(_get(r, 'lock8_out', 'amp'), 3)}  "
                          f"aniso {_fmt(_get(r, 'aniso', 'median'))}  seam {_fmt(_get(r, 'seam', 'ratio'))}  bnd col {_fmt(_get(r, 'bnd', 'col'))} "
                          f"ori {_fmt(_get(r, 'bnd', 'ori'), 3)} logE {_fmt(_get(r, 'bnd', 'logE'))}")
    if not rows:
        raise SystemExit(f"no case folders with mixed.png + mask.npy under {args.roots}")

    print("\nper run (tex = periodic prompts; bnd = face prompts; lock8/aniso = all prompts)")
    print(f"{'run':<34}{'tex ratio':>10}{'lock8 core':>11}{'out':>7}{'aniso':>7}{'seam':>7}{'bnd col':>9}{'ori':>7}{'logE':>7}  streams")
    for run in sorted({r["run"] for r in rows}):
        rr = [r for r in rows if r["run"] == run]
        per = [r for r in rr if r["case"].startswith("periodic")] or rr
        face = [r for r in rr if r["case"].startswith("face")] or rr
        print(f"{run:<34}{_fmt(_mean(_get(r, 'tex', 'ratio') for r in per)):>10}"
              f"{_fmt(_mean(_get(r, 'lock8_core', 'amp') for r in rr), 3):>11}{_fmt(_mean(_get(r, 'lock8_out', 'amp') for r in rr), 3):>7}"
              f"{_fmt(_mean(_get(r, 'aniso', 'median') for r in rr)):>7}{_fmt(_mean(_get(r, 'seam', 'ratio') for r in face)):>7}"
              f"{_fmt(_mean(_get(r, 'bnd', 'col') for r in face)):>9}"
              f"{_fmt(_mean(_get(r, 'bnd', 'ori') for r in face), 3):>7}{_fmt(_mean(_get(r, 'bnd', 'logE') for r in face)):>7}"
              f"  {'yes' if all(r['streams'] for r in rr) else 'no'}")

    eps_runs = sorted({r["run"] for r in rows if r.get("eps")})
    if eps_runs:
        print("\nX3: HR noise estimate in the core (mean over cases). band density, white = 1; lag1, white = 0")
        print(f"{'run':<34}{'step':>5}{'b0':>8}{'b1':>8}{'b2':>8}{'b3':>8}{'lag1':>8}")
        for run in eps_runs:
            rr = [r for r in rows if r["run"] == run and r.get("eps")]
            n = min(len(r["eps"]) for r in rr)
            for j in sorted({0, n // 2, n - 1}):
                vals = [np.mean([r["eps"][j][k] for r in rr]) for k in ("b0", "b1", "b2", "b3", "lag1")]
                print(f"{run:<34}{j:>5}" + "".join(f"{v:8.3f}" for v in vals))
            lw = [_get(r, "last_write_lock", "P8", "amp") for r in rr]
            if any(v is not None for v in lw):
                print(f"{run:<34}  last write Dec(x+gd)-Dec(x): grid lock P8 amp {_mean(lw):.3f}")

    if args.csv:
        keys = ["root", "run", "case", "streams", "tex_ratio", "tex_p_in", "tex_p_out", "lock8_core", "lock8_out",
                "aniso_median", "seam_ratio", "bnd_col", "bnd_ori", "bnd_logE"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r["root"], r["run"], r["case"], r["streams"], _get(r, "tex", "ratio"), _get(r, "tex", "p_in"),
                            _get(r, "tex", "p_out"), _get(r, "lock8_core", "amp"), _get(r, "lock8_out", "amp"),
                            _get(r, "aniso", "median"), _get(r, "seam", "ratio"), _get(r, "bnd", "col"),
                            _get(r, "bnd", "ori"), _get(r, "bnd", "logE")])
        print(f"\nper-case rows -> {args.csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="output roots, e.g. ./phaseQ_out/main ../src_out")
    ap.add_argument("--runs", nargs="*", default=None, help="only run folders containing one of these substrings")
    ap.add_argument("--csv", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    main(ap.parse_args())
