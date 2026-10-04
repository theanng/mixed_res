"""
mask_patterns.py — hand-written HR masks for the cross-scale communication hunt (Track A).

Same HR area across all patterns (token budget fixed), only the GEOMETRY varies, so the only
thing that changes is how much of the HR region sits next to an LR boundary. That quantity ---
boundary_fraction --- is the x-axis: if cross-scale communication is lossy, boundary_excess
should grow with it.

  contiguous : one compact blob            -> lowest boundary density (baseline)
  bisecting  : a band cutting the structure -> a clean cut through periodic/structural content
  strips     : alternating thin strips      -> medium-high boundary density
  scattered  : many small patches           -> highest boundary density

Masks are on the LR latent grid (ht x wt), True = HR.
"""

import torch
import torch.nn.functional as F


def make_mask(pattern, ht, wt, hr_ratio=0.3, strip_w=None, device="cpu"):
    """[1,1,ht,wt] bool HR mask (True=HR), ~hr_ratio area, in the requested geometry.

    strip_w (LR-token units) sets the characteristic HR width for the WIDTH SWEEP:
      * strips   : HR strip is `strip_w` rows tall (spaced to keep ~hr_ratio area)
      * scattered: HR blocks are `strip_w` x strip_w
    Note the rendered HR region is 2x this in HR tokens (each LR cell -> a 2x2 HR block),
    so strip_w=1,2,4 -> HR width 2,4,8 tokens (spanning the ~2r starvation threshold)."""
    m = torch.zeros(ht, wt, dtype=torch.bool)
    if pattern == "full":                         # REF: ~uniform HR (thin LR margin so n_lr>0;
        m[:] = True                               # the pipeline needs some LR tokens to exist)
        m[-2:, :] = False
    elif pattern == "contiguous":                 # compact blob ~ the DeepGaze/saliency status quo
        sh = max(1, int(round((hr_ratio ** 0.5) * ht)))
        sw = max(1, int(round((hr_ratio ** 0.5) * wt)))
        y0, x0 = (ht - sh) // 2, (wt - sw) // 2
        m[y0:y0 + sh, x0:x0 + sw] = True
    elif pattern == "bisecting":                  # one band cutting through the structure
        w = max(1, int(round(hr_ratio * wt)))
        x0 = (wt - w) // 2
        m[:, x0:x0 + w] = True
    elif pattern == "strips":                     # width sweep: strips `strip_w` rows tall
        sw = max(1, int(strip_w)) if strip_w else max(1, int(round(1.0 / hr_ratio)))
        period = max(sw + 1, int(round(sw / hr_ratio)))
        for y in range(0, ht, period):
            m[y:y + sw, :] = True
    elif pattern == "scattered":                  # width sweep: strip_w x strip_w blocks
        b = max(1, int(strip_w)) if strip_w else 2
        step = max(b + 1, int(round(b / (hr_ratio ** 0.5))))
        for y in range(0, ht, step):
            for x in range(0, wt, step):
                m[y:y + b, x:x + b] = True
    elif pattern == "corner":                     # small HR core tucked in the top-left corner so
        sh = max(2, int(round((hr_ratio ** 0.5) * ht)))  # the MIDDLE stays LR for a centred patch
        sw = max(2, int(round((hr_ratio ** 0.5) * wt)))
        m[:sh, :sw] = True
    else:
        raise ValueError(f"unknown mask pattern: {pattern}")
    return m.view(1, 1, ht, wt).to(device)


def _sal_peaks(si, n, radius, rel=0.5):
    """Up to n saliency maxima (y, x) at least `radius` cells apart (Chebyshev), each >= rel * max."""
    work = F.avg_pool2d(si[None, None], 3, 1, 1, count_include_pad=False)[0, 0].clone()
    top, peaks = work.max(), []
    H, W = work.shape
    for _ in range(n):
        v, idx = work.flatten().max(0)
        if v < rel * top:
            break
        y, x = divmod(int(idx), W)
        peaks.append((y, x))
        work[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -float("inf")
    return peaks


def make_sal_mask(pattern, sal, hr_ratio=0.1, strip_w=None):
    """Saliency-placed HR masks on the LR grid. sal: [B,1,H,W] saliency -> [B,1,H,W] bool.

      salblock : top bxb blocks of a fixed b-grid by mean saliency (grid-aligned)
      salcut   : the salient region (top 2*hr_ratio cells), each part cut VERTICALLY through its
                 saliency peak; HR = the right half -> one clean seam through each face/eye
      salcheck : the salient region (top 2*hr_ratio cells) AND a bxb checkerboard -> many seams,
                 but only on the face, so E&R / native bands stay local instead of covering the image
      saleyes  : isolated bxb blocks centred on saliency peaks (>=1 LR cell apart), up to hr_ratio
                 area -> small HR islands on the eyes/mouth, LR everywhere else"""
    B, _, H, W = sal.shape
    b = max(1, int(strip_w)) if strip_w else 2
    out = torch.zeros(B, 1, H, W, dtype=torch.bool, device=sal.device)
    yy, xx = torch.meshgrid(torch.arange(H, device=sal.device), torch.arange(W, device=sal.device), indexing="ij")
    for i in range(B):
        si = sal[i, 0].float()
        if pattern == "salblock":
            s = F.avg_pool2d(si[None, None], b, stride=b, ceil_mode=True)[0, 0]
            k = max(1, int(round(hr_ratio * s.numel())))
            blk = torch.zeros(s.numel(), dtype=torch.bool, device=sal.device)
            blk[s.flatten().topk(k).indices] = True
            blk = blk.view_as(s).repeat_interleave(b, 0).repeat_interleave(b, 1)
            out[i, 0] = blk[:H, :W]
            continue
        if pattern in ("salcut", "salcheck"):
            k = max(1, int(round(2 * hr_ratio * H * W)))
            R = torch.zeros(H * W, dtype=torch.bool, device=sal.device)
            R[si.flatten().topk(k).indices] = True
            R = R.view(H, W)
            if pattern == "salcheck":
                out[i, 0] = R & ((yy // b + xx // b) % 2 == 0)
            else:
                peaks = _sal_peaks(si, n=4, radius=max(4, H // 5))
                py = torch.tensor([p[0] for p in peaks], device=sal.device, dtype=torch.float32)
                px = torch.tensor([p[1] for p in peaks], device=sal.device, dtype=torch.float32)
                d = (yy[..., None] - py) ** 2 + (xx[..., None] - px) ** 2        # [H,W,P]
                own_x = px[d.argmin(-1)]                                          # x of the nearest peak
                out[i, 0] = R & (xx.float() >= own_x)
            continue
        if pattern == "saleyes":
            n = max(1, int(round(hr_ratio * H * W / (b * b))))
            S = F.avg_pool2d(si[None, None], b, stride=1)[0, 0]                 # block score at each top-left
            for _ in range(n):
                v, idx = S.flatten().max(0)
                if not torch.isfinite(v):
                    break
                y0, x0 = divmod(int(idx), S.shape[1])
                out[i, 0, y0:y0 + b, x0:x0 + b] = True
                S[max(0, y0 - b):y0 + b + 1, max(0, x0 - b):x0 + b + 1] = -float("inf")   # keep >=1 cell gap
            continue
        raise ValueError(f"unknown saliency mask pattern: {pattern}")
    return out


def boundary_fraction(mask):
    """Fraction of tokens with a 4-neighbour of the opposite class (HR vs LR). Contiguous -> low,
    scattered -> high. mask: [...,H,W] bool -> float."""
    m = mask.reshape(mask.shape[-2], mask.shape[-1]).bool()
    d = torch.zeros_like(m)
    ud = m[:-1, :] != m[1:, :]
    lr = m[:, :-1] != m[:, 1:]
    d[:-1, :] |= ud; d[1:, :] |= ud
    d[:, :-1] |= lr; d[:, 1:] |= lr
    return d.float().mean().item()
