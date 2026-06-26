"""
seam_spectral.py — spectral seam diagnosis for mixed-resolution generation (v3).

Why this exists: an appearance-similarity test ("how different do the two sides look")
is anti-correlated with the failure we care about. Leopard-fur beside grass looks
different but is fine; moss beside moss looks identical but is the WORST seam. The seam
is a BAND-LIMIT difference, not a content difference, so we measure the band, with a
matched filter to the octave the half-rate LR grid physically cannot represent.

Pixel-space, lr_factor=2 => pixel Nyquist f_N=0.5 c/px, LR Nyquist f_lr=0.25.
  octave band  (f_lr, f_N]   : the energy that MUST be missing on the LR side -> the seam.
  low band     (~0, f_lr]    : the content band E&R operates on.

Readouts (each answers a different question):
  SSS               octave-energy step across the boundary, z-scored against a WITHIN-HR null
                    (subtracts "these are just different textures"). The headline number.
  octave_profile    E_oct vs signed distance to the boundary: step (seam) vs ramp (graded),
                    and the transition WIDTH.
  low_edge_excess   content discontinuity along the boundary in the LOW band -> what E&R fixes.
  strip_spectra     HR-strip vs LR-strip radial power spectra; the gap in the octave is the seam,
                    content-robust because you read the tail SHAPE, not the texture identity.
  phase_jump        for periodic content: misalignment of the dominant pattern across the seam
                    (octave energy is blind to moire/phase).

The E&R ablation figure: low_edge_excess should COLLAPSE with E&R while SSS is UNCHANGED ->
proof E&R operates on the wrong frequency. Validated below on synthetic seams.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy import ndimage


# ------------------------------------------------------------------ primitives
def to_luminance(img):
    img = np.asarray(img, float)
    if img.ndim == 3 and img.shape[-1] >= 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img if img.ndim == 2 else img[..., 0]


def octave_bands(lr_factor=2):
    f_n = 0.5
    f_lr = 0.5 / lr_factor
    return dict(f_n=f_n, f_lr=f_lr, octave=(f_lr, f_n), low=(2.0 / 1, f_lr))  # low lo set per-image


def _radial_freq(h, w):
    fy = fftfreq(h)[:, None]
    fx = fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def bandpass(gray, f_lo, f_hi):
    """Ideal radial band-pass (f_lo, f_hi] in cycles/pixel. Returns the band-passed signal."""
    F = fft2(gray)
    r = _radial_freq(*gray.shape)
    F[~((r > f_lo) & (r <= f_hi))] = 0.0
    return np.real(ifft2(F))


def energy_map(gray, f_lo, f_hi, smooth=8.0):
    """Local energy in the band: |bandpass|^2 smoothed to a per-pixel map."""
    bp = bandpass(gray, f_lo, f_hi)
    return ndimage.gaussian_filter(bp ** 2, smooth)


def radial_power_spectrum(patch, nbins=48):
    patch = np.asarray(patch, float) - np.mean(patch)
    win = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    P = np.abs(np.fft.fftshift(fft2(patch * win))) ** 2
    h, w = P.shape
    y, x = np.indices((h, w))
    rr = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2) / (min(h, w) / 2) * 0.5
    bins = np.linspace(0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(rr.ravel(), bins) - 1, 0, nbins - 1)
    flat = P.ravel()
    out = np.array([flat[idx == b].mean() if (idx == b).any() else 0.0 for b in range(nbins)])
    return 0.5 * (bins[:-1] + bins[1:]), out


# ------------------------------------------------------------ boundary geometry
def _normals(mask, smooth=1.5):
    m = ndimage.gaussian_filter(np.asarray(mask, float), smooth)
    gy, gx = np.gradient(m)
    n = np.hypot(gx, gy) + 1e-9
    return gy / n, gx / n, (m > 0.35) & (m < 0.65)          # unit normal into HR + boundary band


def _sample_pairs(E, mask, offset, n=300, seed=0, within=False):
    """Sample paired energy values either ACROSS the boundary (HR side vs LR side) or, if
    `within`, between two points 2*offset apart both INSIDE the HR region (the null)."""
    H, W = E.shape
    mask = np.asarray(mask, bool)
    ny, nx, band = _normals(mask)
    rng = np.random.default_rng(seed)
    out_a, out_b = [], []
    if within:
        zone = ndimage.binary_erosion(mask, iterations=max(2, offset // 8))
        ys, xs = np.where(zone)
        if len(ys) == 0:
            return np.empty(0), np.empty(0)
        order = rng.permutation(len(ys))
        for i in order:
            y0, x0 = ys[i], xs[i]
            th = rng.uniform(0, np.pi)
            uy, ux = np.sin(th), np.cos(th)
            ay, ax = round(y0 + offset * uy), round(x0 + offset * ux)
            by, bx = round(y0 - offset * uy), round(x0 - offset * ux)
            if not (0 <= ay < H and 0 <= ax < W and 0 <= by < H and 0 <= bx < W):
                continue
            if mask[ay, ax] and mask[by, bx]:
                out_a.append(E[ay, ax]); out_b.append(E[by, bx])
            if len(out_a) >= n:
                break
    else:
        ys, xs = np.where(band)
        if len(ys) == 0:
            return np.empty(0), np.empty(0)
        order = rng.permutation(len(ys))
        for i in order:
            y0, x0 = ys[i], xs[i]
            hy, hx = round(y0 + offset * ny[y0, x0]), round(x0 + offset * nx[y0, x0])
            ly, lx = round(y0 - offset * ny[y0, x0]), round(x0 - offset * nx[y0, x0])
            if not (0 <= hy < H and 0 <= hx < W and 0 <= ly < H and 0 <= lx < W):
                continue
            if mask[hy, hx] and not mask[ly, lx]:
                out_a.append(E[hy, hx]); out_b.append(E[ly, lx])
            if len(out_a) >= n:
                break
    return np.array(out_a), np.array(out_b)


# ------------------------------------------------------------------- the metrics
def seam_spectral_step(img, mask, lr_factor=2, offset=40, smooth=8.0, n=300):
    """SSS: octave-energy step across the boundary, z-scored vs a within-HR null.
    offset must exceed the decode feather (~28 px) so the strips are clean interior."""
    g = to_luminance(img)
    f_lr, f_n = 0.5 / lr_factor, 0.5
    E = energy_map(g, f_lr, f_n, smooth)
    e_hr, e_lr = _sample_pairs(E, mask, offset, n, within=False)
    if len(e_hr) < 8:
        return None
    cross = np.log(e_hr + 1e-9) - np.log(e_lr + 1e-9)       # per-location octave step
    a, b = _sample_pairs(E, mask, offset, n, within=True)   # within-HR null
    null = (np.log(a + 1e-9) - np.log(b + 1e-9)) if len(a) >= 8 else np.array([0.0])
    sss = float((cross.mean() - null.mean()) / (null.std() + 1e-9))
    return dict(sss=sss, octave_step=float(cross.mean()), octave_ratio=float(np.exp(-cross.mean())),
                null_std=float(null.std()), n=len(cross))


def low_edge_excess(img, mask, lr_factor=2, offset=40, smooth=4.0, n=300):
    """Content discontinuity across the boundary in the LOW band (the density/level shift E&R
    targets), measured like SSS: |L_HR - L_LR| across the seam vs |L_a - L_b| within HR.
    ~1 = no content seam; >1 = a low-frequency shift across the boundary."""
    g = to_luminance(img)
    # keep everything below the LR Nyquist INCLUDING region-scale (only global DC, r==0, is dropped),
    # so a regional density/brightness shift across the boundary is captured.
    L = ndimage.gaussian_filter(bandpass(g, 0.0, 0.5 / lr_factor), smooth)
    hr, lr = _sample_pairs(L, mask, offset, n, within=False)
    if len(hr) < 8:
        return None
    cross = np.abs(hr - lr)
    a, b = _sample_pairs(L, mask, offset, n, within=True)
    null = np.abs(a - b) if len(a) >= 8 else np.array([cross.mean()])
    return float(cross.mean() / (null.mean() + 1e-9))


def octave_profile(img, mask, lr_factor=2, smooth=8.0, maxd=120, binw=8):
    """E_oct vs signed distance to the boundary (+ inside HR). Returns the profile and the
    transition WIDTH (10%->90% of the step) — step (narrow) vs graded ramp (wide)."""
    g = to_luminance(img)
    E = energy_map(g, 0.5 / lr_factor, 0.5, smooth)
    mask = np.asarray(mask, bool)
    sdf = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    edges = np.arange(-maxd, maxd + binw, binw)
    d = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([np.log(E[(sdf >= edges[i]) & (sdf < edges[i + 1])].mean() + 1e-9)
                     if ((sdf >= edges[i]) & (sdf < edges[i + 1])).any() else np.nan
                     for i in range(len(edges) - 1)])
    far = maxd * 0.6
    lr_lvl = np.nanmean(prof[d <= -far]); hr_lvl = np.nanmean(prof[d >= far])
    step = float(hr_lvl - lr_lvl)
    width = float("nan")
    if step > 0.2:
        lo_t, hi_t = lr_lvl + 0.1 * step, lr_lvl + 0.9 * step
        cross = d[(prof >= lo_t) & (prof <= hi_t) & ~np.isnan(prof)]
        if len(cross):
            width = float(cross.max() - cross.min())
    return dict(d=d, profile=prof, step=step, width=width)


def strip_spectra(img, mask, lr_factor=2, offset=40, patch=64, max_pairs=60, seed=0):
    """HR-strip vs LR-strip radial power spectra (the reference-free seam figure)."""
    g = to_luminance(img)
    H, W = g.shape
    ny, nx, band = _normals(mask)
    mask = np.asarray(mask, bool)
    rng = np.random.default_rng(seed)
    ys, xs = np.where(band)
    h = patch // 2
    hrs, lrs = [], []
    for i in rng.permutation(len(ys)):
        y0, x0 = ys[i], xs[i]
        hy, hx = round(y0 + offset * ny[y0, x0]), round(x0 + offset * nx[y0, x0])
        ly, lx = round(y0 - offset * ny[y0, x0]), round(x0 - offset * nx[y0, x0])
        if not (h <= hy < H - h and h <= hx < W - h and h <= ly < H - h and h <= lx < W - h):
            continue
        if mask[hy, hx] and not mask[ly, lx]:
            hrs.append(g[hy - h:hy + h, hx - h:hx + h]); lrs.append(g[ly - h:ly + h, lx - h:lx + h])
        if len(hrs) >= max_pairs:
            break
    if len(hrs) < 4:
        return None
    f = radial_power_spectrum(hrs[0])[0]
    P_hr = np.mean([radial_power_spectrum(p)[1] for p in hrs], axis=0)
    P_lr = np.mean([radial_power_spectrum(p)[1] for p in lrs], axis=0)
    return dict(freqs=f, P_hr=P_hr, P_lr=P_lr, f_lr=0.5 / lr_factor)


def phase_jump(img, mask, lr_factor=2, offset=40, patch=64, max_pairs=40, seed=0):
    """For periodic content: 1 - normalized cross-correlation of the octave-band HR/LR strips
    across the seam. ~0 = pattern continues; ->1 = pattern misaligns/aliases (moire)."""
    g = to_luminance(img)
    sp = strip_spectra(img, mask, lr_factor, offset, patch, max_pairs, seed)
    if sp is None:
        return None
    # only meaningful if the HR side has a clear spectral peak (a periodic pattern)
    f, P = sp["freqs"], sp["P_hr"]
    band = f >= 0.5 / lr_factor
    if not band.any() or P[band].max() < 3 * np.median(P[f > 0.05] + 1e-12):
        return dict(phase_jump=float("nan"), periodic=False)
    H, W = g.shape
    ny, nx, bnd = _normals(mask)
    mask = np.asarray(mask, bool)
    rng = np.random.default_rng(seed)
    ys, xs = np.where(bnd)
    h = patch // 2
    cc = []
    for i in rng.permutation(len(ys)):
        y0, x0 = ys[i], xs[i]
        hy, hx = round(y0 + offset * ny[y0, x0]), round(x0 + offset * nx[y0, x0])
        ly, lx = round(y0 - offset * ny[y0, x0]), round(x0 - offset * nx[y0, x0])
        if not (h <= hy < H - h and h <= hx < W - h and h <= ly < H - h and h <= lx < W - h):
            continue
        if not (mask[hy, hx] and not mask[ly, lx]):
            continue
        a = bandpass(g[hy - h:hy + h, hx - h:hx + h], 0.5 / lr_factor, 0.5)
        b = bandpass(g[ly - h:ly + h, lx - h:lx + h], 0.5 / lr_factor, 0.5)
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-9
        cc.append(float((a * b).sum() / denom))
        if len(cc) >= max_pairs:
            break
    return dict(phase_jump=float(1.0 - np.mean(cc)) if cc else float("nan"), periodic=True)


def scale_shift(img, mask, lr_factor=2, fmin=0.04, offset=40, patch=64, max_pairs=60):
    """Dominant-SCALE shift across the seam — the readout SSS is blind to.

    SSS measures the octave-tail energy DEFICIT (resolution seam). The grouping/sinc envelope
    instead fixes the same element rendered at two SIZES (big/small stone, mismatched brick
    courses) — a shift in the spectral CENTROID, not the tail, so SSS stays flat while it moves.

    scale_shift_oct = log2( centroid_freq(HR strip) / centroid_freq(LR strip) ).
    ~0 = both sides render at the same scale (harmonized); >0 = HR finer than LR (the jump)."""
    if np.asarray(mask).shape != to_luminance(img).shape:
        from PIL import Image
        mask = np.array(Image.fromarray(np.asarray(mask)).resize(to_luminance(img).shape[::-1])) > 0
    sp = strip_spectra(img, mask, lr_factor, offset, patch, max_pairs)
    if sp is None:
        return None
    f, Ph, Pl = sp["freqs"], sp["P_hr"], sp["P_lr"]
    band = f >= fmin

    def centroid(P):
        w = np.clip(P[band], 0, None)
        return float((f[band] * w).sum() / (w.sum() + 1e-12))

    ch, cl = centroid(Ph), centroid(Pl)
    return dict(scale_shift_oct=float(np.log2((ch + 1e-9) / (cl + 1e-9))),
                centroid_hr=ch, centroid_lr=cl,
                peak_hr=float(f[band][np.argmax(Ph[band])]), peak_lr=float(f[band][np.argmax(Pl[band])]))


def analyze(img, mask, lr_factor=2):
    """Full reference-free seam analysis."""
    if np.asarray(mask).shape != to_luminance(img).shape:
        from PIL import Image
        mask = np.array(Image.fromarray(np.asarray(mask)).resize(to_luminance(img).shape[::-1])) > 0
    out = {}
    s = seam_spectral_step(img, mask, lr_factor)
    if s is None:
        return None
    out.update(s)
    out["low_edge_excess"] = low_edge_excess(img, mask, lr_factor)
    out["octave_width"] = octave_profile(img, mask, lr_factor)["width"]
    pj = phase_jump(img, mask, lr_factor)
    out["phase_jump"] = pj["phase_jump"] if pj else float("nan")
    ss = scale_shift(img, mask, lr_factor)
    out["scale_shift_oct"] = ss["scale_shift_oct"] if ss else float("nan")
    return out


# ----------------------------------------------- aggregate over seam_out/{er,noer}
def diagnose_folder(root="seam_out", tags=("er", "noer"), lr_factor=2):
    """Aggregate by category across arbitrary tag folders (e.g. er / noer / group_er / sinc_er).
    Prints one block per metric: SSS (resolution seam), lowEx (content seam), scale (the
    scale-jump grouping/sinc target). Expects root/<tag>/<category>_<i>/{mixed.png,mask.npy,meta.json}."""
    import glob, json, os
    from collections import defaultdict
    from PIL import Image
    have = [t for t in tags if os.path.isdir(os.path.join(root, t))]
    res = {}
    for tag in have:
        rows = defaultdict(list)
        for d in sorted(glob.glob(os.path.join(root, tag, "*"))):
            if not os.path.exists(os.path.join(d, "mask.npy")):
                continue
            img = np.asarray(Image.open(os.path.join(d, "mixed.png")).convert("RGB"), float) / 255
            mask = np.load(os.path.join(d, "mask.npy"))
            meta = json.load(open(os.path.join(d, "meta.json")))
            a = analyze(img, mask, lr_factor)
            if a:
                rows[meta["category"]].append(a)
        res[tag] = rows
    cats = sorted({c for t in have for c in res[t]})
    for key, lab, absval in [("sss", "SSS (resolution seam)", False),
                             ("low_edge_excess", "lowEx (content seam)", False),
                             ("scale_shift_oct", "|scale| octaves (scale jump)", True)]:
        print(f"\n[{lab}]      " + "".join(f"{t:>11}" for t in have))
        for c in cats:
            cells = []
            for t in have:
                v = [abs(r[key]) if absval else r[key] for r in res[t].get(c, [])]
                cells.append(f"{np.nanmean(v):>11.2f}" if v else f"{'-':>11}")
            print(f"{c:<24}" + "".join(cells))
    return res


# ============================ VALIDATION ============================
def _one_over_f(n=512, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fftshift(fft2(rng.standard_normal((n, n))))
    y, x = np.indices((n, n)); r = np.hypot(x - n // 2, y - n // 2); r[n // 2, n // 2] = 1
    img = np.real(ifft2(np.fft.ifftshift(F / r ** alpha)))
    return (img - img.min()) / (img.max() - img.min())


def _square(n, frac=0.5):
    yy, xx = np.indices((n, n))
    return (np.abs(xx - n / 2) < frac * n / 2) & (np.abs(yy - n / 2) < frac * n / 2)


def _lowpass_img(g, f_hi):
    return bandpass(g, 0.0, f_hi) + g.mean()


def _row(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<48} {detail}")
    return ok


def run_validation():
    r = []; n = 512; m = _square(n, 0.5); f_lr = 0.25
    base = _one_over_f(n, 0.8, 3)

    print("1) band-pass isolates the octave")
    yy, xx = np.indices((n, n))
    sine_oct = np.cos(2 * np.pi * 0.33 * xx)                 # 0.33 c/px -> in octave (0.25,0.5]
    e_oct = energy_map(sine_oct, f_lr, 0.5).mean()
    e_low = energy_map(sine_oct, 0.02, f_lr).mean()
    r.append(_row("octave sine -> energy in octave, not low", e_oct > 50 * (e_low + 1e-9),
                  f"oct/low={e_oct/(e_low+1e-9):.0f}"))

    print("2) SSS fires on a SPECTRAL seam, stays silent on CONTENT drift / no seam")
    spec = base.copy(); spec[~m] = _lowpass_img(base, f_lr)[~m]       # LR side loses the octave
    drift = base.copy(); drift[~m] = _one_over_f(n, 0.8, 9)[~m]       # different full-res texture
    s_spec = seam_spectral_step(spec, m)["sss"]
    s_drift = seam_spectral_step(drift, m)["sss"]
    s_none = seam_spectral_step(base, m)["sss"]
    r.append(_row("spectral seam -> SSS large", s_spec > 4, f"SSS={s_spec:.1f}"))
    r.append(_row("content drift -> SSS ~0 (no false positive)", abs(s_drift) < 2, f"SSS={s_drift:.1f}"))
    r.append(_row("no seam -> SSS ~0", abs(s_none) < 2, f"SSS={s_none:.1f}"))
    r.append(_row("spectral SSS >> drift SSS (specificity)", s_spec > 3 * abs(s_drift) + 2,
                  f"{s_spec:.1f} vs {s_drift:.1f}"))

    print("3) low_edge_excess = the CONTENT seam (a low-freq density shift, what E&R fixes)")
    # a smooth low-frequency density gradient on the LR side = the shift E&R targets.
    # NOTE: two random textures with the SAME low-freq statistics are NOT a content seam.
    yy, xx = np.indices((n, n))
    shift = 0.18 * (xx / n + yy / n)           # real-amplitude low-freq density gradient
    content = base.copy(); content[~m] = (base + shift)[~m]   # full spectrum (octave intact) + low shift
    le_spec = low_edge_excess(spec, m)         # low band identical both sides -> ~1
    le_cont = low_edge_excess(content, m)      # low-freq jump at boundary -> >1
    r.append(_row("spectral seam -> low_edge_excess ~1", le_spec < 1.4, f"{le_spec:.2f}"))
    r.append(_row("content shift -> low_edge_excess elevated", le_cont > 1.5, f"{le_cont:.2f}"))
    r.append(_row("content shift -> SSS ~0 (octave intact both sides)",
                  abs(seam_spectral_step(content, m)["sss"]) < 2,
                  f"SSS={seam_spectral_step(content, m)['sss']:.1f}"))

    print("4) E&R ablation signature: low collapses, octave (SSS) unchanged")
    # 'noer' = spectral seam WITH the low-band shift; 'er' = same seam, low-band harmonized.
    noer = base.copy(); noer[~m] = (_lowpass_img(base, f_lr) + shift)[~m]
    er = base.copy();   er[~m] = _lowpass_img(base, f_lr)[~m]
    le_er, le_no = low_edge_excess(er, m), low_edge_excess(noer, m)
    sss_er, sss_no = seam_spectral_step(er, m)["sss"], seam_spectral_step(noer, m)["sss"]
    r.append(_row("E&R lowers low_edge_excess", le_er < le_no - 0.15, f"er={le_er:.2f} noer={le_no:.2f}"))
    r.append(_row("E&R leaves SSS ~unchanged", abs(sss_er - sss_no) < 0.4 * max(sss_er, sss_no),
                  f"er={sss_er:.1f} noer={sss_no:.1f}"))

    print("5) octave profile: a sharp seam is a STEP")
    prof = octave_profile(spec, m)
    r.append(_row("spectral seam -> positive octave step", prof["step"] > 1.0, f"step={prof['step']:.1f}"))

    print("6) scale_shift: same element at two SIZES -> octave jump; same scale -> ~0")
    def _tex(scale, seed):                                    # band-limited texture, peak freq ~ 1/scale
        rng = np.random.default_rng(seed); f0 = 1.0 / scale
        F = np.fft.fftshift(fft2(rng.standard_normal((n, n))))
        yy2, xx2 = np.indices((n, n)); rr = np.hypot(xx2 - n // 2, yy2 - n // 2) / (n / 2) * 0.5
        F *= np.exp(-((rr - f0) ** 2) / (2 * (0.4 * f0) ** 2))
        g = np.real(ifft2(np.fft.ifftshift(F))); return (g - g.min()) / (g.max() - g.min() + 1e-9)
    mism = _tex(6, 1).copy(); mism[~m] = _tex(12, 2)[~m]      # HR fine (6px) vs LR coarse (12px) = 2x
    same = _tex(6, 1).copy(); same[~m] = _tex(6, 3)[~m]       # same scale both sides
    ss_m = scale_shift(mism, m)["scale_shift_oct"]; ss_s = scale_shift(same, m)["scale_shift_oct"]
    r.append(_row("2x size mismatch -> ~1 octave", 0.5 < ss_m < 1.5, f"oct={ss_m:+.2f}"))
    r.append(_row("same scale -> ~0 octaves", abs(ss_s) < 0.35, f"oct={ss_s:+.2f}"))
    r.append(_row("mismatch >> harmonized (specificity)", ss_m > 3 * abs(ss_s), f"{ss_m:+.2f} vs {ss_s:+.2f}"))

    print(f"\n{sum(r)}/{len(r)} checks passed.")
    return all(r)


if __name__ == "__main__":
    run_validation()
