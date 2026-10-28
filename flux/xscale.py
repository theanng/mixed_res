"""
xscale.py -- cross-scale operators for the SCR relay (SCR_method_spec.md §2.3).

    R_up   (512 -> 1024):  up(x_lr)   [B,1024,64] -> [B,4096,64]
    R_down (1024 -> 512):  down(x_hr) [B,4096,64] -> [B,1024,64]

All tensors are PACKED transformer-space latents (16 ch x 2x2 latent px per token), as everywhere
else in the pipeline.

    mode      R_up                                   R_down                          trained parts
    bilinear  latent bilinear x2 (+mean align)       latent avg-pool 2 (+mean align)  none  (legacy relay)
    haar      latent nearest x2                      latent avg-pool 2               none  (exact pyramid, D U = I)
    taef1     taef1 decode -> bicubic x2 -> encode   decode -> area /2 -> encode     off-the-shelf tiny AE
    vae       FLUX VAE decode -> bicubic x2 -> enc   decode -> area /2 -> encode     the model's own VAE
    learned   CRPA bi_upscaler                       CRPA lt_downscaler              E&R resizers (comparison only)

`bilinear` and `learned` go through `pipe._resize_x0_to_x0` unchanged, so the legacy relay reproduces
bit-for-bit. `taef1` / `vae` run in fp32 (the relay uses a DIFFERENCE of two encodings; bf16 is too
coarse for that). Pixel conventions match flux/resizer_align.py. Only torch is imported here, so the
module loads without diffusers (CPU unit tests).
"""

import copy

import torch
import torch.nn.functional as F

MODES = ("bilinear", "haar", "taef1", "vae", "learned")
PIXEL_MODES = ("taef1", "vae")


def _up_px(img):
    return F.interpolate(img, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)


def _dn_px(img):
    return F.interpolate(img, scale_factor=0.5, mode="area")


class XScale:
    """R_up / R_down for one mode. Build it with build_xscale(pipe, mode) so the fp32 autoencoder
    copies are made once per pipeline, not once per image."""

    def __init__(self, pipe, mode, ae=None):
        if mode not in MODES:
            raise ValueError(f"unknown xscale mode {mode!r}; expected one of {MODES}")
        self.pipe, self.mode, self.ae = pipe, mode, ae
        self.crop = None            # (y0, y1, x0, x1) in LR-latent px; only used by the pixel modes

    # ---------------------------------------------------------------- packing (same as _resize_x0_to_x0)
    def _unpack(self, x, s):
        p = self.pipe
        return p._unpack_latents(x, p._h * s, p._w * s, p.vae_scale_factor)

    def _pack(self, lt, s):
        return self.pipe._pack_lt_w_scale(lt, s)

    # ---------------------------------------------------------------- optional region crop (--scr_crop)
    def set_crop(self, core_mask_lr, margin_tokens=2):
        """Restrict the expensive pixel-space ops to the bounding box of the HR region plus a margin
        (2 LR tokens = 64 px at 1024, wider than the 28 px decode feather). Token-aligned, so LR and HR
        crops stay on their token grids. Outside the box the cheap linear operator is used; the relay
        only reads the region, so that never changes the correction. core_mask_lr: [B,1,ht,wt] bool."""
        if core_mask_lr is None:
            self.crop = None
            return
        m = core_mask_lr.reshape(-1, *core_mask_lr.shape[-2:]).any(0)
        ys, xs = torch.nonzero(m, as_tuple=True)
        if ys.numel() == 0:
            self.crop = None
            return
        ht, wt = m.shape
        ty0 = max(0, int(ys.min()) - margin_tokens); ty1 = min(ht, int(ys.max()) + 1 + margin_tokens)
        tx0 = max(0, int(xs.min()) - margin_tokens); tx1 = min(wt, int(xs.max()) + 1 + margin_tokens)
        self.crop = (2 * ty0, 2 * ty1, 2 * tx0, 2 * tx1)        # 1 LR token = 2x2 LR-latent px

    # ---------------------------------------------------------------- pixel-space round trips (fp32)
    def _dec(self, lt):
        if self.mode == "taef1":
            return self.ae.decode(lt).sample
        cfg = self.ae.config
        return self.ae.decode(lt / cfg.scaling_factor + cfg.shift_factor, return_dict=False)[0]

    def _enc(self, img):
        if self.mode == "taef1":
            return self.ae.encode(img).latents
        cfg = self.ae.config
        return (self.ae.encode(img).latent_dist.mode() - cfg.shift_factor) * cfg.scaling_factor

    def _px_up(self, lt):
        return self._enc(_up_px(self._dec(lt)))

    def _px_down(self, lt):
        return self._enc(_dn_px(self._dec(lt)))

    # ---------------------------------------------------------------- public operators
    @torch.no_grad()
    def up(self, x):
        """[B,1024,64] -> [B,4096,64]. Accepts any batch (the relay batches u_a, u_b)."""
        if self.mode == "bilinear":
            return self.pipe._resize_x0_to_x0(x, 2.0, "bilinear", ori_scale=1)[0]
        if self.mode == "learned":
            return self.pipe._resize_x0_to_x0(x, 2.0, "bi", ori_scale=1)[0]
        lt = self._unpack(x, 1).float()
        if self.mode == "haar":
            out = lt.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        elif self.crop is None:
            out = self._px_up(lt)
        else:
            y0, y1, x0, x1 = self.crop
            out = F.interpolate(lt, scale_factor=2, mode="bilinear", align_corners=False)
            out[..., 2 * y0:2 * y1, 2 * x0:2 * x1] = self._px_up(lt[..., y0:y1, x0:x1].contiguous())
        return self._pack(out, 2).to(x.dtype)

    @torch.no_grad()
    def down(self, x):
        """[B,4096,64] -> [B,1024,64]."""
        if self.mode == "bilinear":
            return self.pipe._resize_x0_to_x0(x, 0.5, "avgpool", ori_scale=2)[0]
        if self.mode == "learned":
            return self.pipe._resize_x0_to_x0(x, 0.5, "", ori_scale=2)[0]
        lt = self._unpack(x, 2).float()
        if self.mode == "haar":
            out = F.avg_pool2d(lt, 2)
        elif self.crop is None:
            out = self._px_down(lt)
        else:
            y0, y1, x0, x1 = self.crop
            out = F.avg_pool2d(lt, 2)
            out[..., y0:y1, x0:x1] = self._px_down(lt[..., 2 * y0:2 * y1, 2 * x0:2 * x1].contiguous())
        return self._pack(out, 1).to(x.dtype)


def build_xscale(pipe, mode):
    """Cached XScale for this pipeline. The fp32 autoencoder copies are created once and reused."""
    cache = getattr(pipe, "_xscale_cache", None)
    if cache is None:
        cache = pipe._xscale_cache = {}
    if mode not in cache:
        ae = None
        if mode == "taef1":
            ae = copy.deepcopy(pipe.tvae).float().eval()          # ~1M params
        elif mode == "vae":
            ae = copy.deepcopy(pipe.vae).float().eval()           # FLUX VAE in fp32, ~0.3 GB
        cache[mode] = XScale(pipe, mode, ae=ae)
    return cache[mode]
