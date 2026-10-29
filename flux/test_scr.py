"""
test_scr.py -- unit / regression tests for SCR (SCR_method_spec.md §6). CPU, no model, no diffusers.

The real pipeline methods are loaded from flux/pipeline_flux_2stg.py by source (the module itself
imports diffusers), bound to a stub pipeline with a mock transformer and a mock Euler scheduler.

    T1  haar round trip                down(up(x)) == x
    T2  exact range-null replacement   haar + band none + gain 1: D(x_H') = x_L on the region and
                                       (I - U D)(x_H' - x_H) = 0;  z_H' moves by exactly (1 - s') d_H
    T3  promotion whiteness            --scr_promote preserve gives white HR noise, D(eps_H) = eps_L / 2
    T4  legacy reproduction            XScale(bilinear) == _resize_x0_to_x0, and the new _relay_step with
                                       all SCR flags at default == the OLD _relay_step (git 2026d21), bitwise
    T6  gain-0 regression              gain 0 leaves the HR result = the plain HR scheduler step
    T7  crpa context                   forward_mixed runs with the transformer's native_ctx 'off', restored after
    T8  band filters                   gauss does not overshoot a region-masked constant; brick rings (W3)
    T9  region crop                    pixel-mode ops with --scr_crop match the full op on the region
    T10 image metrics                  aniso ~1 on isotropic texture, >> 1 on vertical streaks; E_pix rises
                                       when dec_hr departs from the upsampled dec_lr
    T11 matched fake noise             --scr_fake_noise matched feeds the LR pass (1-s_L) D(x_H) + s_L 2 D(eps_H)
    T12 eps whiteness (Phase Q X3)     white eps -> band densities ~1, lag1 ~0 on small and salblock-like masks;
                                       LR-scale content relabelled as noise raises b0/b1 and lowers b3
    T13 offline metrics (Phase Q X5)   grid_lock / row_period / tex_ratio / seam_jump on synthetic images
    T5  (GPU, --gpu)                   taef1 / vae round-trip error and shift on real latents

    python -m flux.test_scr            # from mixed_res_dit/
    python -m flux.test_scr --gpu --device cuda:0
"""

import argparse
import ast
import contextlib
import functools
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from .xscale import XScale, build_xscale

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE_SRC = os.path.join(HERE, "pipeline_flux_2stg.py")
HELPERS_SRC = os.path.join(HERE, "utils", "helpers.py")
LEGACY_REV = "2026d21"
METHODS = ("_pack_lt_w_scale", "_resize_x0_to_x0", "_add_noise_at_t", "_lowpass_packed", "_gauss_packed",
           "_scr_xscale", "_scr_flags", "_scr_preserve_init", "_relay_step", "_scr_image_metrics", "_scr_eps_stats")


# ----------------------------------------------------------------------------------------------- loading
def _defs(src, names, cls=None):
    tree = ast.parse(src)
    body = tree.body
    if cls is not None:
        body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    return [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]


def _namespace():
    ns = {"torch": torch, "F": F, "np": np, "functools": functools, "contextlib": contextlib, "time": time,
          "build_xscale": build_xscale}
    helpers = _defs(open(HELPERS_SRC).read(), {"align_latent_mean_shift_flexible", "transform_base_mask"})
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "helpers", "exec"), ns)
    timed = _defs(open(PIPE_SRC).read(), {"_scr_timed"})
    exec(compile(ast.Module(body=timed, type_ignores=[]), "scr_timed", "exec"), ns)
    return ns


def _pack(lat, b, c, h, w):
    return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack(lat, height, width, vsf):
    b, n, c = lat.shape
    height = 2 * (int(height) // (vsf * 2)); width = 2 * (int(width) // (vsf * 2))
    lat = lat.view(b, height // 2, width // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return lat.reshape(b, c // 4, height, width)


def make_stub(relay_src=None):
    """Stub pipeline class holding the real methods. relay_src: source to take _relay_step from."""
    ns = _namespace()
    src = open(PIPE_SRC).read()
    fns = _defs(src, set(METHODS) - ({"_relay_step"} if relay_src else set()), cls="FluxPipeline")
    if relay_src:
        fns += _defs(relay_src, {"_relay_step"}, cls="FluxPipeline")
    exec(compile(ast.Module(body=fns, type_ignores=[]), "pipe_methods", "exec"), ns)
    attrs = {f.name: ns[f.name] for f in fns}
    attrs.update(_pack_latents=staticmethod(_pack), _unpack_latents=staticmethod(_unpack))
    Stub = type("PipeStub", (), attrs)
    p = Stub()
    p._h = p._w = 512; p.vae_scale_factor = 8; p._c_lt = 16
    p.joint_attention_kwargs = None
    f32 = lambda f: (lambda x: f(x.float()).to(x.dtype))
    p.bi_upscaler = f32(lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                        + 0.05 * torch.tanh(F.interpolate(x, scale_factor=2, mode="nearest")))
    p.lt_upscaler = p.bi_upscaler
    p.lt_downscaler = f32(lambda x: F.avg_pool2d(x, 2) + 0.05 * torch.tanh(x[..., ::2, ::2]))
    p.transformer = MockTF()
    return p


# ----------------------------------------------------------------------------------------------- mocks
class MockTF:
    """Deterministic velocity field. lr_target: if set, LR-canvas calls return v with x0 = lr_target."""

    def __init__(self):
        self.native_ctx = "relay"
        self.lr_target = None
        self.calls = []

    @contextlib.contextmanager
    def cache_context(self, name):
        yield

    @staticmethod
    def vel(h):
        g = torch.Generator().manual_seed(int(h.shape[1]))
        bias = torch.randn(h.shape[1:], generator=g)
        return (0.5 * torch.tanh(h.float()) + 0.1 * bias).to(h.dtype)

    def __call__(self, hidden_states, timestep, joint_attention_kwargs=None, **kw):
        self.calls.append(("std", hidden_states.shape[1], dict(joint_attention_kwargs or {})))
        if hidden_states.shape[1] == 1024 and self.lr_target is not None:
            s = float(timestep.reshape(-1)[0])
            return ((hidden_states.float() - self.lr_target.float()) / s).to(hidden_states.dtype),
        return self.vel(hidden_states),

    def forward_mixed(self, hidden_states_lr, hidden_states_hr, mask_lr_tokens, mask_lr_keep_tokens, **kw):
        self.calls.append(("mixed", self.native_ctx))
        hr_tok = mask_lr_tokens.repeat_interleave(2, -2).repeat_interleave(2, -1).reshape(-1)
        idx = hr_tok.nonzero().squeeze(1)
        out = hidden_states_hr.clone()
        out[:, idx] = self.vel(hidden_states_hr[:, idx])
        return hidden_states_lr.clone(), out


class MockSched:
    """Euler on rectified flow, same arithmetic and casting as the repo's scheduler."""

    def __init__(self, sigmas):
        self.sigmas, self.i = sigmas, 0

    def step(self, v, t, z, return_dict=False):
        s, sn = self.sigmas[self.i], self.sigmas[self.i + 1]
        self.i += 1
        zf = z.to(torch.float32)
        return (zf + (sn - s) * v).to(v.dtype), (zf - s * v).to(v.dtype)


def geometry(dtype=torch.float32, seed=0):
    core = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    core[..., 10:14, 8:12] = True                     # 4x4-cell block
    core[..., 20:22, 20:22] = True                    # 2x2-cell island (salblock-sized)
    seq_lr = core.float().reshape(1, 1024, 1)
    seq_hr = core.float().repeat_interleave(2, -2).repeat_interleave(2, -1).reshape(1, 4096, 1)
    hr_idx = seq_hr[0, :, 0].nonzero().squeeze(1)
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dtype)
    return dict(core=core, seq_lr=seq_lr, seq_hr=seq_hr, hr_idx=hr_idx,
                z_lr=r(1, 1024, 64), z_hr=r(1, 4096, 64), x_lr=r(1, 1024, 64), noise=r(1, 1024, 64))


SIG_L = [0.74, 0.65, 0.55, 0.45]
SIG_H = [0.82, 0.74, 0.65, 0.55]


def run_relay(p, geo, n_steps=1, **flags):
    for k, v in flags.items():
        setattr(p, k, v)
    p._scr_cache, p._scr_log = {}, None
    sl, sh = MockSched(SIG_L), MockSched(SIG_H)
    z_lr, z_hr, x_lr = geo["z_lr"], geo["z_hr"], geo["x_lr"]
    outs = []
    import inspect
    extra = {"core_hr_mask_LR": geo["core"]} if "core_hr_mask_LR" in inspect.signature(p._relay_step).parameters else {}
    for j in range(n_steps):
        t_lr = torch.tensor(SIG_L[j] * 1000); t_hr = torch.tensor(SIG_H[j] * 1000)
        nxt = torch.tensor(SIG_H[j + 1] * 1000)
        o = p._relay_step(z_lr, z_hr, x_lr, t_lr, t_hr, t_lr.reshape(1), t_hr.reshape(1), nxt, sl, sh,
                          geo["noise"], geo["seq_lr"], geo["seq_hr"], geo["hr_idx"], None, None,
                          torch.zeros(1, 512, 8), None, torch.zeros(1024, 3), torch.zeros(4096, 3), 2, **extra)
        outs.append(o)
        z_lr, z_hr, x_lr = o[0], o[1], o[2]
    return outs


def U(lt):
    return lt.repeat_interleave(2, -2).repeat_interleave(2, -1)


# ----------------------------------------------------------------------------------------------- tests
def t1_haar_roundtrip():
    p = make_stub()
    xs = XScale(p, "haar")
    x = torch.randn(2, 1024, 64)
    err = (xs.down(xs.up(x)) - x).abs().max().item()
    assert err < 1e-6, err
    return f"max err {err:.1e}"


def t2_exact_replacement():
    p = make_stub()
    geo = geometry()
    target = torch.randn(1, 1024, 64, generator=torch.Generator().manual_seed(7))
    p.transformer.lr_target = target
    # HR estimate the step will produce (before the relay): replicate pass + scheduler
    pred = torch.zeros_like(geo["z_hr"])
    pred[:, geo["hr_idx"]] = MockTF.vel(geo["z_hr"][:, geo["hr_idx"]])
    hr_new0, hr_x0_0 = MockSched(SIG_H).step(pred, None, geo["z_hr"])
    lr_new, hr_new, lr_x0, hr_x0 = run_relay(p, geo, scr_xscale="haar", scr_band="none", relay_gain=1.0)[0]
    assert torch.allclose(lr_x0, target, atol=1e-5)
    D = lambda x: F.avg_pool2d(_unpack(x, 1024, 1024, 8), 2)
    unp_lr = _unpack(target, 512, 512, 8)
    m_lr = U(geo["core"].float()).expand(1, 16, 64, 64) > 0.5            # region on the LR-latent grid
    e_range = (D(hr_x0) - unp_lr)[m_lr].abs().max().item()
    dH = _unpack(hr_x0 - hr_x0_0, 1024, 1024, 8)
    e_null = (dH - U(F.avg_pool2d(dH, 2))).abs().max().item()
    m_hr = geo["seq_hr"][0, :, 0] > 0.5
    e_out = (hr_x0 - hr_x0_0)[:, ~m_hr].abs().max().item()
    e_z = ((hr_new - hr_new0) - (1 - SIG_H[1]) * (hr_x0 - hr_x0_0)).abs().max().item()
    assert e_range < 1e-5 and e_null < 1e-5 and e_out == 0.0 and e_z < 1e-5, (e_range, e_null, e_out, e_z)
    return f"range {e_range:.1e}  null {e_null:.1e}  outside {e_out:.1e}  z-propagation {e_z:.1e}"


def t3_promotion_whiteness():
    p = make_stub()
    xs = XScale(p, "haar")
    g = torch.Generator().manual_seed(3)
    n = 4
    x_lr = torch.randn(n, 1024, 64, generator=g)
    eps_true = torch.randn(n, 1024, 64, generator=g)
    s_l, s_h = 0.738, 0.818
    z_lr = (1 - s_l) * x_lr + s_l * eps_true
    xi = torch.randn(n, 4096, 64, generator=g)
    _, _, eps_h = p._scr_preserve_init(z_lr, x_lr, xi, s_l * 1000, s_h * 1000, xs)
    e = _unpack(eps_h, 1024, 1024, 8)                                    # [n,16,128,128]
    blocks = e.unfold(-2, 2, 2).unfold(-2, 2, 2).reshape(-1, 4)          # 2x2 blocks -> 4-vectors
    cov = torch.cov(blocks.T)
    off = (cov - torch.diag(torch.diag(cov))).abs().max().item()
    dia = (torch.diag(cov) - 1).abs().max().item()
    e_pool = (F.avg_pool2d(e, 2) - 0.5 * _unpack(eps_true, 512, 512, 8)).abs().max().item()
    assert off < 0.02 and dia < 0.02 and e_pool < 1e-5, (off, dia, e_pool)
    return f"{blocks.shape[0]} blocks: |offdiag| {off:.4f}  |diag-1| {dia:.4f}  D(eps_H) - eps_L/2 {e_pool:.1e}"


def _legacy_src():
    try:
        return subprocess.run(["git", "show", f"{LEGACY_REV}:flux/pipeline_flux_2stg.py"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def t4_legacy_reproduction():
    p = make_stub()
    x = torch.randn(1, 1024, 64).to(torch.bfloat16)
    h = torch.randn(1, 4096, 64).to(torch.bfloat16)
    b = XScale(p, "bilinear")
    assert torch.equal(b.up(x), p._resize_x0_to_x0(x, 2.0, "bilinear", ori_scale=1)[0])
    assert torch.equal(b.down(h), p._resize_x0_to_x0(h, 0.5, "avgpool", ori_scale=2)[0])
    old = _legacy_src()
    if old is None:
        return "XScale(bilinear) bitwise OK; git history unavailable, old-relay comparison skipped"
    configs = [dict(relay_resampler="bilinear", relay_gain=1.0, relay_text_bias=0.0, relay_cut=1.0),
               dict(relay_resampler="bilinear", relay_gain=0.0, relay_text_bias=0.0, relay_cut=1.0),
               dict(relay_resampler="learned", relay_gain=1.0, relay_text_bias="auto", relay_cut=1.0),
               dict(relay_resampler="bilinear", relay_gain=0.5, relay_text_bias=0.0, relay_cut=0.5)]
    for cfg in configs:
        geo = geometry(dtype=torch.bfloat16)
        new_out = run_relay(make_stub(), geo, n_steps=3, **cfg)
        old_out = run_relay(make_stub(relay_src=old), geo, n_steps=3, **cfg)
        for a, b_ in zip(new_out, old_out):
            assert all(torch.equal(u, v) for u, v in zip(a, b_)), cfg
    return f"XScale(bilinear) bitwise OK; new _relay_step == {LEGACY_REV} _relay_step bitwise, {len(configs)} configs x 3 steps"


def t6_gain0():
    p = make_stub()
    geo = geometry()
    pred = torch.zeros_like(geo["z_hr"])
    pred[:, geo["hr_idx"]] = MockTF.vel(geo["z_hr"][:, geo["hr_idx"]])
    hr_new0, hr_x0_0 = MockSched(SIG_H).step(pred, None, geo["z_hr"])
    for mode in ("bilinear", "haar"):
        _, hr_new, _, hr_x0 = run_relay(p, geo, scr_xscale=mode, scr_band="none", relay_gain=0.0)[0]
        assert torch.equal(hr_new, hr_new0) and torch.equal(hr_x0, hr_x0_0), mode
    return "HR result == plain HR scheduler step (bilinear, haar)"


def t7_crpa_context():
    p = make_stub()
    geo = geometry()
    _, hr_new, _, _ = run_relay(p, geo, scr_ctx="crpa", scr_xscale="haar", scr_band="none")[0]
    mixed = [c for c in p.transformer.calls if c[0] == "mixed"]
    assert mixed and all(c[1] == "off" for c in mixed), mixed
    assert p.transformer.native_ctx == "relay"
    out = geo["seq_hr"][0, :, 0] < 0.5
    assert torch.equal(hr_new[:, out], geo["z_hr"][:, out])            # tokens outside the region untouched
    return "forward_mixed saw native_ctx='off', restored to 'relay'; tokens outside the region unchanged"


def t8_band_filters():
    p = make_stub()
    geo = geometry()
    m = geo["seq_hr"].expand(1, 4096, 64)
    g = (p._gauss_packed(m.clone(), 2) * geo["seq_hr"])
    br = (p._lowpass_packed(m.clone(), 2) * geo["seq_hr"])
    inside = geo["seq_hr"][0, :, 0] > 0.5
    gmin, gmax = g[:, inside].min().item(), g[:, inside].max().item()
    bmin, bmax = br[:, inside].min().item(), br[:, inside].max().item()
    assert gmax <= 1 + 1e-5 and gmin >= -1e-5, (gmin, gmax)
    return f"constant correction on the region: gauss in [{gmin:.2f}, {gmax:.2f}], brick in [{bmin:.2f}, {bmax:.2f}]"


class _LocalAE:
    """Stand-in for a pixel autoencoder: local decode (x8 nearest) / encode (8x8 average), 16 'colour' ch."""

    class _O:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def decode(self, lt):
        return self._O(sample=0.1 * lt.repeat_interleave(8, -2).repeat_interleave(8, -1))

    def encode(self, img):
        return self._O(latents=10.0 * F.avg_pool2d(img, 8))


def t9_crop():
    p = make_stub()
    geo = geometry()
    full = XScale(p, "taef1", ae=_LocalAE())
    crop = XScale(p, "taef1", ae=_LocalAE())
    crop.set_crop(geo["core"])
    x_lr = torch.randn(1, 1024, 64)
    x_hr = torch.randn(1, 4096, 64)
    m_hr = geo["seq_hr"][0, :, 0] > 0.5
    m_lr = geo["seq_lr"][0, :, 0] > 0.5
    e_up = (full.up(x_lr) - crop.up(x_lr))[:, m_hr].abs().max().item()
    e_dn = (full.down(x_hr) - crop.down(x_hr))[:, m_lr].abs().max().item()
    assert e_up < 1e-5 and e_dn < 1e-5, (e_up, e_dn)
    return f"crop {crop.crop} (LR-latent px): region error up {e_up:.1e}, down {e_dn:.1e}"


def t10_metrics():
    p = make_stub()
    geo = geometry()
    g = torch.Generator().manual_seed(5)
    dec_lr = F.avg_pool2d(torch.randn(1, 3, 1024, 1024, generator=g), 2).clamp(-1, 1)
    smooth = F.interpolate(dec_lr, scale_factor=2, mode="bicubic", align_corners=False)
    iso = p._scr_image_metrics(smooth, dec_lr, geo["core"])
    streak = smooth.clone()
    streak[..., 1::4] += 0.8                                            # vertical lines every 4 px
    st = p._scr_image_metrics(streak, dec_lr, geo["core"])
    assert 0.5 < iso["aniso_mean"] < 2.0 and st["aniso_mean"] > 4 * iso["aniso_mean"], (iso, st)
    assert st["E_pix"] > iso["E_pix"], (iso, st)
    return (f"smooth: E_pix {iso['E_pix']:.3f}, aniso {iso['aniso_mean']:.2f};  "
            f"streaked: E_pix {st['E_pix']:.3f}, aniso {st['aniso_mean']:.1f}")


class _ExactHRTF(MockTF):
    """HR calls return the exact velocity for a known (x, eps): v = eps - x, so x_H = x and eps_H = eps."""

    def __init__(self, x_h, eps_h):
        super().__init__()
        self.x_h, self.eps_h, self.lr_in = x_h, eps_h, None

    def __call__(self, hidden_states, timestep, img_ids=None, joint_attention_kwargs=None, **kw):
        if hidden_states.shape[1] == 1024:
            self.lr_in = hidden_states.clone()
            return MockTF.vel(hidden_states),
        idx = self.idx
        return (self.eps_h[:, idx] - self.x_h[:, idx]),


def t11_matched_noise():
    """--scr_fake_noise matched: the fake fed to the LR pass is (1 - s_L) D(x_H) + s_L * 2 D(eps_H) in the region."""
    p = make_stub()
    geo = geometry()
    g = torch.Generator().manual_seed(11)
    x_h = torch.randn(1, 4096, 64, generator=g)
    eps_h = torch.randn(1, 4096, 64, generator=g)
    s_h, s_l = SIG_H[0], SIG_L[0]
    geo["z_hr"] = (1 - s_h) * x_h + s_h * eps_h
    tf = _ExactHRTF(x_h, eps_h)
    tf.idx = geo["hr_idx"]
    p.transformer = tf
    run_relay(p, geo, scr_xscale="haar", scr_band="none", relay_gain=0.0, scr_fake_noise="matched")
    D = lambda a: _pack(F.avg_pool2d(_unpack(a, 1024, 1024, 8), 2), 1, 16, 64, 64)
    want = (1 - s_l) * D(x_h) + s_l * 2.0 * D(eps_h)
    m = geo["seq_lr"][0, :, 0] > 0.5
    err = (tf.lr_in[:, m] - want[:, m]).abs().max().item()
    out = (tf.lr_in[:, ~m] - geo["z_lr"][:, ~m]).abs().max().item()
    noise = 2.0 * D(eps_h)[:, m]
    assert err < 1e-4 and out == 0.0, (err, out)
    return f"region err {err:.1e}, outside untouched, matched noise std {noise.std():.3f}"


def t12_eps_stats():
    """X3: white eps reads ~1 in every band whatever the mask; LR-scale content relabelled as noise does not."""
    p = make_stub()
    geo = geometry()
    g = torch.Generator().manual_seed(12)
    sb = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    rng = np.random.default_rng(0)
    for _ in range(26):                                                  # salblock-like: 2x2-cell blocks, ~10%
        y, x = rng.integers(0, 15, 2) * 2
        sb[..., y:y + 2, x:x + 2] = True
    res = {}
    for name, core in (("small", geo["core"].expand(8, -1, -1, -1)), ("sb", sb.expand(8, -1, -1, -1))):
        white = torch.randn(8, 4096, 64, generator=g)
        w = p._scr_eps_stats(white, core)
        assert all(0.85 < w[f"b{k}"] < 1.15 for k in range(4)) and abs(w["lag1"]) < 0.02, (name, w)
        res[name] = w
    low = F.interpolate(torch.randn(8, 16, 32, 32, generator=g), size=(128, 128), mode="bilinear", align_corners=False)
    r = p._scr_eps_stats(torch.randn(8, 4096, 64, generator=g) + _pack(low, 8, 16, 128, 128), geo["core"].expand(8, -1, -1, -1))
    assert r["b0"] > 4 and r["b1"] > 3 and r["b3"] < 0.85 and r["lag1"] > 0.15, r
    return (f"white small {[res['small'][f'b{k}'] for k in range(4)]} lag1 {res['small']['lag1']:+.3f}; "
            f"white sb {[res['sb'][f'b{k}'] for k in range(4)]}; relabelled b0 {r['b0']:.1f} b1 {r['b1']:.1f} "
            f"b3 {r['b3']:.2f} lag1 {r['lag1']:.2f}")


def t13_offline_metrics():
    """X5: the offline metrics on synthetic images with known answers."""
    from scipy import ndimage
    from .scr_metrics import grid_lock, row_period, tex_ratio, seam_jump
    rng = np.random.default_rng(13)
    noise = rng.standard_normal((256, 256)) * 10
    m = np.ones((256, 256), bool)
    free = grid_lock(noise, m, 8)["amp"]
    locked = grid_lock(noise + np.tile(rng.standard_normal((8, 8)) * 5, (32, 32)), m, 8)["amp"]
    assert free < 0.05 and locked > 0.3, (free, locked)
    yy = np.arange(128)[:, None] * np.ones((1, 128))
    per = row_period(np.sin(2 * np.pi * yy / 20) + 0.1 * rng.standard_normal((128, 128)))[0]
    assert abs(per - 20) < 0.5, per
    H = 768
    yy = np.arange(H)[:, None] * np.ones((1, H))
    mask = np.zeros((H, H), bool)
    mask[256:512, 256:512] = True
    tex = tex_ratio(100 * np.sin(2 * np.pi * yy / 12) + rng.standard_normal((H, H)),
                    100 * np.sin(2 * np.pi * yy / 30) + rng.standard_normal((H, H)), mask)["ratio"]
    assert abs(tex - 0.4) < 0.03, tex
    H = 1024
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal((H, H)), 12) for _ in range(3)], -1)
    f = (f - f.min()) / (f.max() - f.min()) * 255
    mask = np.zeros((H, H), bool)
    mask[384:640, 384:640] = True
    same, shifted = seam_jump(f, f, mask)["ratio"], seam_jump(np.roll(f, 40, 1), f, mask)["ratio"]
    assert abs(same - 1) < 0.05 and shifted > 3, (same, shifted)
    return (f"grid lock P8 free {free:.3f} / locked {locked:.2f}; period {per:.2f} (20); tex ratio {tex:.3f} (0.4); "
            f"seam same {same:.3f}, 40-px shift {shifted:.1f}")


def t5_gpu(device, images, n=8):
    """taef1 / vae round trip on real latents: |down(up(x)) - x| / |x| and the low-frequency shift."""
    import glob
    from diffusers import AutoencoderKL, AutoencoderTiny
    from PIL import Image
    from .resizer_align import shift, encode
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae",
                                        torch_dtype=torch.float32).to(device)
    p = make_stub()
    p.vae = vae
    p.tvae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.float32).to(device)
    paths = sorted(glob.glob(images))[:n]
    if not paths:
        return f"no images match {images}"
    rows = {"taef1": [], "vae": []}
    for path in paths:
        img = Image.open(path).convert("RGB")
        s = min(img.size)
        img = img.crop(((img.width - s) // 2, (img.height - s) // 2, (img.width + s) // 2, (img.height + s) // 2))
        L = encode(vae, img, 512, device, torch.float32)
        x = _pack(L, 1, 16, 64, 64)
        for mode in rows:
            xs = build_xscale(p, mode)
            y = xs.down(xs.up(x))
            a, b = _unpack(y, 512, 512, 8)[0].cpu().numpy(), L[0].cpu().numpy()
            rows[mode].append((float(np.linalg.norm(a - b) / np.linalg.norm(b)), shift(a, b) * 16))
    return "  ".join(f"{m}: rel err {np.mean([r[0] for r in v]):.3f}, shift "
                     f"({np.mean([r[1][0] for r in v]):+.1f}, {np.mean([r[1][1] for r in v]):+.1f}) px"
                     for m, v in rows.items())


def main(args):
    tests = [("T1 haar round trip", t1_haar_roundtrip), ("T2 exact range-null replacement", t2_exact_replacement),
             ("T3 promotion whiteness", t3_promotion_whiteness), ("T4 legacy reproduction", t4_legacy_reproduction),
             ("T6 gain-0 regression", t6_gain0), ("T7 crpa context", t7_crpa_context),
             ("T8 band filters", t8_band_filters), ("T9 region crop", t9_crop), ("T10 image metrics", t10_metrics),
             ("T11 matched fake noise", t11_matched_noise), ("T12 eps whiteness", t12_eps_stats),
             ("T13 offline metrics", t13_offline_metrics)]
    failed = 0
    for name, fn in tests:
        try:
            print(f"PASS  {name}: {fn()}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    if args.gpu:
        print(f"INFO  T5 pixel-mode round trip: {t5_gpu(args.device, args.images)}")
    else:
        print("SKIP  T5 pixel-mode round trip (needs --gpu)")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images", default="../er_out/*/*/mixed.png")
    main(ap.parse_args())
