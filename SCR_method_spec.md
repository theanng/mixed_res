# SCR: Scale-Consistent Relay (implementation spec, v1)

**Repo:** `theanng/mixed_res` @ `2026d21` (Oct 28 2026). Code lives in `flux/`, run as `python -m flux.<module>`.
**Base model:** FLUX.1-dev. Stage 1 is all-LR at 512; stage 2 runs LR (512 canvas) and HR (1024 region) together.
**Goal:** training-free, persistent mixed-resolution generation. The HR region must agree with its LR surroundings, with no learned resizers.

This document is the hand-off to the implementer. It covers four things:
- what the current `relay` does and what is wrong with it (§1);
- the method (§2);
- file-by-file changes and flags (§3–4);
- diagnostics, tests and the experiment order with predictions (§5–7).

Keep every new option **off by default**, so old commands reproduce bit-for-bit.

---

## 0. TL;DR for the implementer

1. Add a cross-scale operator module `flux/xscale.py` with modes `bilinear` | `haar` | `taef1` | `vae` | `learned`. The two pixel-space modes reuse the exact conventions already validated in `flux/resizer_align.py`.
2. Extend `_relay_step` in `flux/pipeline_flux_2stg.py` in four ways:
   - (a) a **context mode** for the HR pass: `none` = current; `crpa` = HR queries also see the real LR tokens through PMA/CRPA;
   - (b) the operator from step 1 for both the fake-LR build and the relay correction;
   - (c) a choice of band filter: `none` | `gauss` | `brick` (`brick` = current);
   - (d) per-step diagnostics.
3. Add an optional **noise-preserving stage-2 entry**, and an option to skip the LR re-noise at stage-2 start.
4. Optionally replace the final-decode fill outside the HR core (currently `repeat` latents) with the operator from step 1.
5. Add CPU unit tests (§6). Then run experiments E0→E7 (§7) in order, because each one gates the next.

---

## 1. Current relay: what it does, what is right, what is wrong

### 1.1 What `_relay_step` does (one stage-2 step)

These are the lines in `pipeline_flux_2stg.py::_relay_step` (~L176–243), with notation from §2.1:

1. **HR pass.** The HR region tokens plus text only go through a standard `self.transformer(...)` forward with native 1024 ids at `t_hr` (optional text logit bias). This gives `hr_new` and `hr_x0` = x̂_H.
2. **Fake LR.** `fake_x0 = avgpool(hr_x0 ⊕ up(lr_x0_prev))` (avgpool for the bilinear option, learned `lt_down` otherwise). It is re-noised at σ_L with the fixed `stage2_lr_noise` and pasted into the LR canvas region.
3. **LR pass.** A native 512 forward on the complete canvas gives x̂_L.
4. **Relay.** `corr = m_H · LP_brick( up(x̂_L) − up(b) )`, where `b` = x̂_L outside the region and `fake_x0` inside it. Then `hr_x0 += corr` and `hr_new += (1−σ'_H)·corr`.

### 1.2 What is already right (keep it)

| Property | Why it is correct |
|---|---|
| **Delta form** `up(after) − up(before)` | It vanishes when the LR pass leaves the fake unchanged, so the fixed point is consistency itself. It cancels the resampler's bias to first order: R↑(a)−R↑(b) = J(b)(a−b)+O(‖a−b‖²). With U = nearest, D = avg and no low-pass, it is **exactly** DDNM's range–null replacement (unit-tested in §6, T2). It is a replacement, not double counting. |
| **Correction applied in x̂₀ space and propagated as (1−σ')·corr into z** | Euler on rectified flow gives z' = (1−σ')x̂₀ + σ'ε̂. Changing x̂₀ by δ therefore changes z' by exactly (1−σ')δ. This avoids the different time parametrizations of the two schedulers (shift 3.0 vs 1.8776). |
| **Fake LR re-noised from x̂₀ at σ_L** | The noise level is matched to the LR canvas by construction. Pooling the noisy z_H would halve the noise std instead. |
| **Two native streams, each with its own `temb` and native integer RoPE ids** | Each forward is in-distribution: a valid 512 canvas and a valid 1024 token set. No mixed AdaLN, no cross-scale RoPE. |

### 1.3 What is wrong (ordered by expected impact)

**W1. The HR pass has no image context.** This is the main cause of the vertical streaks.

The HR stream is HR region + text only, which is exactly the Phase J `native_ctx=full` geometry. That geometry produced "stripe garbage", and settled finding #12 says HR windows need far context.

The relay pins only the band at or below the LR Nyquist. The streaks live in the octave above it, which nothing constrains. Measured on the bilinear relay output, Ex/Ey (horizontal over vertical gradient energy per 32-px cell) reaches **12–25** on the streaked cells of the man's face. For comparison, the plain wall is 0.6 and clean skin is about 1.

*To confirm:* run E0 (`--relay_gain 0` gives the same streaks).

**W2. The latent spaces are not scale-equivariant.** This is a systematic bias injected at every step.

avgpool(E₁₀₂₄(I)) ≠ E₅₁₂(down(I)). The repo's own `resizer_align` gives r = 0.82 for box-down against the true 512 latent. If r is a Pearson correlation over latent entries, about 1−r² ≈ 33% of the 512-latent variance is not explained by the pooled 1024 latent. EQ-VAE (ICML 2025) shows the same failure for SD/SDXL/SD3 VAEs, where decoding a downscaled latent degrades badly; FLUX was not tested there, but our r = 0.82 says the same.

Consequences:
- The fake LR is **off the 512 latent manifold**.
- The LR pass partly "repairs" it, so Δ = x̂_L − fake is **non-zero even when content agrees**.
- The relay pushes up(Δ) into HR at every step.

The noise part of the pyramid is exact; see §2.8.

**W3. Brick-wall low-pass on a region-masked signal causes Gibbs ringing.**

`_lowpass_packed` is an ideal rectangle in frequency. Applied to a correction that is non-zero only in the region and then masked, it rings parallel to the region edges with a period of about 4 HR-latent px (32 image px).

Measured: a constant correction on an 8-latent-px-wide window (salblock 2×2 cells) comes out with inside values in [0.71, 1.13], i.e. **±20%**, plus 2.3% energy leaked outside.

**W4. Bilinear does not invert avgpool.**

D∘U_bilinear is the filter [1/8, 3/4, 1/8], with response 0.75 + 0.25·cos(2πg). That is 0.5 at the LR Nyquist in 1-D and 0.25 at the 2-D corner. So only part of the correction at the top of the LR band lands per step. Measured gains at g = 0.0625 / 0.125 / 0.25 / 0.375 / 0.5 are 0.98 / 0.93 / 0.75 / 0.57 / 0.50.

**W5. The final-decode fill outside the core is `repeat` (nearest) latents, and `pad2 = 0`.**

Inside the 28-px tanh feather, `dec_hr` contributes 50% at the boundary and about 12% at 28 px out. In that zone `dec_hr` is the 1024-VAE decode of nearest-upsampled 512 latents, which is off-manifold. The E&R runs did not have this problem because they kept a 2-cell band of real HR latents around the core.

**W6. Minor: the LR canvas is re-noised at stage-2 start.**

`latents_lr = add_noise(latents_lr_x0, stage2_lr_noise, t_lr0)` throws away the stage-1 state z_L at the same σ and redraws its noise. The HR start is also 18% structure + 82% fresh noise (hr_shift 3). Neither is tied to the LR noise realization.

---

## 2. Method

### 2.1 Notation

All latents are **packed** `[B, N, 64]` in transformer latent space:
- z_H: 4096 tokens; z_L: 1024 tokens;
- x̂_H, x̂_L: x̂₀ estimates;
- m_H `[B,4096,1]`, m_L `[B,1024,1]`: HR-core masks (`relay_seq_hr`, `relay_seq_lr`);
- σ_H, σ_L: current sigmas (t/1000); σ'_H, σ'_L: next sigmas (0 at the last step);
- ε_L^fix = `stage2_lr_noise`;
- R↓: HR→LR and R↑: LR→HR are the cross-scale operators (§2.3);
- B: band filter (§2.5); g: `relay_gain`.

### 2.2 Per-step algorithm (replaces the body of `_relay_step`)

```
# 1) HR pass (context mode, §2.4)
v_H            = HRPass(z_H, z_L, t_H, t_L)
x̂_H            = z_H − σ_H v_H ;  z'_H = z_H + (σ'_H − σ_H) v_H          # = sched_hr.step(...)

# 2) fake LR = LR-native view of the current HR estimate
fill_H         = cache.up_after  if available else R↑(x̂_L_prev)          # R↑ of last step's LR estimate
c_H            = m_H ⊙ x̂_H + (1 − m_H) ⊙ fill_H                          # composite so conv receptive fields see context
f_L            = R↓(c_H)
z̃_L            = (1 − m_L) ⊙ z_L + m_L ⊙ [(1 − σ_L) f_L + σ_L ε_L^fix]

# 3) LR pass on the complete native canvas (unchanged)
v_L            = Transformer_512(z̃_L, t_L)
x̂_L            = z̃_L − σ_L v_L ;  z'_L = z̃_L + (σ'_L − σ_L) v_L

# 4) relay (delta form)
b_L            = (1 − m_L) ⊙ x̂_L + m_L ⊙ f_L
u_a, u_b       = R↑(x̂_L), R↑(b_L)                                         # batch the two calls
δ_H            = m_H ⊙ B(u_a − u_b)
x̂_H           += g δ_H ;   z'_H += g (1 − σ'_H) δ_H
cache.up_after = u_a
```

For `haar` and `bilinear` (linear operators), compute `u_a − u_b = R↑(x̂_L − b_L)` with a single call.

### 2.3 Cross-scale operators R↓ / R↑ (`flux/xscale.py`)

| Mode | R↑ (512→1024) | R↓ (1024→512) | Notes |
|---|---|---|---|
| `bilinear` | latent bilinear ×2 (align_corners=False) | latent avgpool 2 | Legacy. Route through `_resize_x0_to_x0(..., "bilinear"/"avgpool")` with its default `align_mean=True` so old runs reproduce exactly. |
| `haar` | latent nearest (`repeat_interleave`) | latent avgpool 2 | Exact latent pyramid (D U = I). The delta relay becomes exact range–null replacement. Wrong signal domain (W2). Ablation only. |
| `taef1` | `tae.encode(up_px(tae.decode(L).sample)).latents` | `tae.encode(dn_px(tae.decode(L).sample)).latents` | **Recommended default.** Copy the lambdas from `resizer_align.py` L115–121. taef1 consumes transformer-space latents directly, as `_tvae_decode` does. `up_px` = bicubic ×2 `.clamp(-1,1)`; `dn_px` = `F.interpolate(scale_factor=0.5, mode="area")`. |
| `vae` | `to_latent(vae, up_px(from_latent(vae, L)))` | `to_latent(vae, dn_px(from_latent(vae, L)))` | Reference, using the model's own VAE. Copy `to_latent` / `from_latent` from `resizer_align.py` L72–78 (scaling/shift handled there). |
| `learned` | `bi_upscaler` | `lt_downscaler` | Legacy E&R resizers. Comparison only; not part of the training-free claim. |

Implementation rules:
- **Signature.** `XScale(pipe, mode)` with methods `up(x_lr_packed)` → `[B,4096,64]` and `down(x_hr_packed)` → `[B,1024,64]`. Unpack with `pipe._unpack_latents(x, pipe._h*s, pipe._w*s, pipe.vae_scale_factor)` and pack with `pipe._pack_lt_w_scale(lt, s)` (s = 1 for LR, 2 for HR), exactly as `_resize_x0_to_x0` does.
- **Precision.** Run `taef1` and `vae` in **fp32**. The relay uses a *difference* of two encodings, and bf16 (~3 significant digits) is too coarse for a δ that can be ~5% of the latent magnitude. Make an fp32 copy at setup: `pipe.tvae_fp32 = copy.deepcopy(pipe.tvae).float()`. For `vae`, an fp32 copy of the FLUX VAE (~0.3 GB) is fine. Cast back to the pipeline dtype on return.
- **Batching.** `up()` must accept batch 2 (u_a and u_b in one call).
- **Region crop (optimization, after correctness).** Decode/encode only the bounding box of `m` plus a 64-px margin, aligned to 16 px. Keep a `--scr_crop` flag, off by default.

Why the pixel-space modes are the principled choice: the final render compares `dec_hr` with `bicubic(dec_lr)`, so consistency is a **pixel-space** condition. With R↓ = E₅₁₂∘down∘Dec₁₀₂₄:
- the fake is an on-manifold 512 latent of the HR content, so Δ ≈ 0 when the content agrees and W2 disappears;
- at the fixed point, Dec₅₁₂(x̂_L) ≈ down(Dec₁₀₂₄(x̂_H)) inside the region. That is exactly the low band the final bicubic-blend needs to match.

These modes are the exact, training-free counterpart of what E&R's learned resizers were trained to imitate.

### 2.4 HR-pass context mode (`--scr_ctx`)

**`none` (current).** Standard `self.transformer(...)` on `latents_hr[:, hr_idx]` + text, with optional `relay_text_bias`. Expected to show W1 streaks.

**`crpa` (recommended).** The HR pass is the existing plain-CRPA mixed forward (`FluxTransformer2DModel.forward_mixed` in `transformer_flux_2t.py`, the two-timestep version), so HR queries get context from the real LR tokens outside the region through PMA. Use only its HR output.

```python
tf = self.transformer
saved = tf.native_ctx
tf.native_ctx = "off"                  # IMPORTANT: forward_mixed changes behaviour when native_ctx != "off"
tf._mixed_mask_cache = None            # once at stage-2 start (masks constant afterwards)
_, noise_pred_hr = tf.forward_mixed(
    hidden_states_lr=latents_lr,                     # current LR canvas state z_L (region tokens excluded below)
    hidden_states_hr=latents_hr,
    mask_lr_tokens=core_hr_mask_LR,                  # [B,1,32,32] bool, HR region
    mask_lr_keep_tokens=~core_hr_mask_LR,            # LR tokens outside the region (plain CRPA, no overlap)
    timestep_lr=timestep_lr / 1000, timestep_hr=timestep_hr / 1000,
    guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids, img_ids_lr=ids_lr, img_ids_hr=ids_hr,
    joint_attention_kwargs=self.joint_attention_kwargs)
tf.native_ctx = saved
hr_new, hr_x0 = sched_hr.step(noise_pred_hr, t_hr, latents_hr, return_dict=False)
```

- Pass `core_hr_mask_LR` (the bchw bool mask) into `_relay_step`; it currently only receives token masks.
- The `pipe.native_ctx` attribute stays `"relay"`. Only the transformer attribute is toggled around this call.
- `--area_scale hr` (+log 4 on LR keys for HR queries) works here unchanged and is a sub-ablation.
- Rationale: the two components fix different failures. PMA supplies **context** (where things are, so no stripes); the relay supplies **consistency** (what the LR region actually becomes, so no P2). §7 E6 is the 2×2 that tests this.

**`lrkv` (optional v2, do not implement first).** HR queries read the LR pass's per-layer K/V (native LR features) at CRPA positions, instead of running a mixed forward. It is cheaper and in-distribution, but needs KV capture hooks and either running the LR pass first or accepting a one-step lag.

### 2.5 Band filter `B` (`--scr_band`)

- `none`: identity. Default for `haar` (keeps exactness) and for `taef1`/`vae` (the pixel-space delta is already band-limited by the bicubic ×2).
- `gauss`: separable Gaussian on the **unpacked** HR latent, σ = 0.75 latent px (gain 0.5 at 0.25 cyc/px), reflect padding, radius 3. Apply **before** the m_H mask.
- `brick`: legacy `_lowpass_packed(..., cut_scale=relay_cut)`. Kept for reproduction only (W3).

### 2.6 Stage-2 entry and schedule

**`--scr_lr_restart off`.** Skip `latents_lr = self._add_noise_at_t(latents_lr_x0, stage2_lr_noise, stage2_start_t_lr)` and keep the stage-1 state. Stage 2 starts at the same sigma index where stage 1 ended (`num_inference_steps = stg1 + stg2`), so this is a pure continuation.

**`--scr_promote preserve`.** A noise-realization-preserving HR start. In unpacked space, using U = nearest and D = avgpool:

```
σ_L0 = t_lr0/1000 ; σ_H0 = t_hr0/1000
ε̂_L  = (z_L − (1 − σ_L0) x̂_L) / σ_L0          # z_L = stage-1 output, x̂_L = latents_lr_x0
ξ    = latents_hr_noise
ε_H  = ½ U(ε̂_L) + (ξ − U(D(ξ)))               # white: cov = ¼·4P + (I − P) = I  (§6, T3)
z_H  = (1 − σ_H0) R↑(x̂_L) + σ_H0 ε_H
```

- Compute ε̂_L from the stage-1 output **before** any LR re-noise line runs, so it is valid whether `--scr_lr_restart` is on or off.
- ε̂_L computed this way equals the model's own noise estimate from the last stage-1 Euler step (algebraically exact).
- The coarse part of the HR noise *is* the LR noise realization, and only the new octave gets fresh noise. This is the latent-space version of SPD's "spectral noise expansion" and RALU's correlated-noise construction.
- It is noise-exact in any VAE; only the signal term R↑(x̂_L) depends on the operator.

**Matched schedule (no code; run flag).** `--hr_shift 3.7552211585` (= 2 × lr_shift).
- From logit σ = ln s + logit t, and because avg-pooling white noise multiplies its std by ½, pooled HR noise equals the LR noise level at **every** step iff s_H = 2 s_L. At stage-2 start this gives σ_H = 0.849 (currently 0.818).
- With s_H = 3, the HR low band is "ahead": its pooled-noise σ is 0.692 vs the LR canvas at 0.738.
- Expected effect is small, because the relay overwrites the HR low band every step. Treat it as a cleanliness ablation.

### 2.7 Final decode fill (`--scr_final_fill`)

In the final block of `__call__`, replace

```python
lt_hr_fake, _ = self._resize_x0_to_x0(latents_lr, hr_scale, "repeat", ori_scale=1, align_mean=False)
```

with `lt_hr_fake = XScale(self, mode).up(latents_lr)` when the flag is `xscale` and native_ctx is `relay`. With `taef1`/`vae`, `dec_hr` outside the core becomes ≈ a VAE round trip of `bicubic(dec_lr)`, so the 28-px feather blends two nearly identical images (fixes W5). This costs one extra call per image.

### 2.8 Theory summary (for the paper and for sanity checks)

1. **Fixed point.** δ_H = 0 ⇔ R↑(x̂_L) = R↑(b_L) on the region ⇔ (R↑ injective there) the LR model leaves R↓(x̂_H) unchanged. So the relay's stationary state is "the HR estimate, viewed at 512 inside the full LR canvas, is a state the LR model accepts". The prof's hypothesis is that HR must see what the LR region turns into; this makes that operational.
2. **Exactness in the Haar case.** With U = nearest, D = avg, B = none: D(x̂_H + U(x̂_L − D x̂_H)) = x̂_L on the region, and (I − UD) x̂_H is unchanged. This holds in one step because the mask is aligned to LR cells (T2).
3. **What is exact across 512/1024 and what is not.**
   - *Exact:* token format (64-d = 16 ch × 2×2 latent px), each stream's integer RoPE grid, each stream's temb, Gaussian noise algebra in latent space (pooling, promotion, matched shift).
   - *Not exact:* the **signal** identification avgpool(E₁₀₂₄ I) = E₅₁₂(down I). The VAE is not scale-equivariant (W2), which is why R↓/R↑ must go through pixel space.
   - A token covers 16×16 px of *its own* image, which is 32×32 px (LR) vs 16×16 px (HR) of the final canvas. Scale consistency comes from the pixel-space down/up, not from the tokens.
4. **Delta-form bias cancellation.** u_a − u_b = J_{R↑}(b_L)(x̂_L − b_L) + O(‖·‖²). A constant operator bias (VAE round-trip error, taef1 vs FLUX encoder mismatch) cancels; only the Jacobian matters.

---

## 3. Code changes, file by file

### 3.1 `flux/xscale.py` (new)
- `class XScale` per §2.3.
- `build_xscale(pipe, mode)` creates and caches fp32 copies (`pipe._xscale_cache[mode]`).

### 3.2 `flux/pipeline_flux_2stg.py`

**`_relay_step`:**
- Add parameter `core_hr_mask_LR`.
- Read new attributes: `scr_ctx`, `scr_xscale`, `scr_band`, `scr_log`.
- Implement §2.2 / §2.4 / §2.5.
- When `scr_xscale is None`, map from the legacy `relay_resampler` (`learned` → `learned`, `bilinear` → `bilinear`) and keep `brick`, so defaults reproduce the current results.
- Keep the text-bias code path for `scr_ctx == "none"`.

**Stage-2 setup (around the `Stage-2 init` block, ~L796–812):** implement `scr_lr_restart` and `scr_promote` (§2.6). Only active when `native_ctx == "relay"`.

**Stage-2 loop, relay branch (~L843–864):**
- Pass `core_hr_mask_LR`.
- Reset `self._scr_cache = {}` at j == 0.
- Set `self.transformer._mixed_mask_cache = None` at j == 0 if `scr_ctx == "crpa"`.

**Final decode (~L958, the `lt_hr_fake` line):** §2.7.

**Diagnostics:** append a per-step dict to `self._scr_log` (§5).

### 3.3 `flux/run_seam.py`
- Add the flags in §4 and assign them to `pipe` next to the existing relay attributes (~L108–111).
- In `save_case`, if `pipe._scr_log` exists, write it into `meta.json` under `"scr"`, together with all SCR flag values.

### 3.4 `flux/test_scr.py` (new, CPU-only, no model)
Covers T1–T5 in §6, with `XScale` linear modes exercised on random tensors through a minimal stub `pipe` that has `_unpack_latents` / `_pack_lt_w_scale`.

---

## 4. Flags (all default to current behaviour)

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--scr_ctx` | `none`, `crpa` | `none` | HR-pass context (§2.4) |
| `--scr_xscale` | `bilinear`, `haar`, `taef1`, `vae`, `learned` | unset → legacy `relay_resampler` | R↓/R↑ operator (§2.3) |
| `--scr_band` | `none`, `gauss`, `brick` | `brick` | Band filter on δ (§2.5) |
| `--scr_promote` | `renoise`, `preserve` | `renoise` | Stage-2 HR entry (§2.6) |
| `--scr_lr_restart` | `on`, `off` | `on` | Re-noise the LR canvas at stage-2 start |
| `--scr_final_fill` | `repeat`, `xscale` | `repeat` | Final-decode HR fill outside the core (§2.7) |
| `--scr_crop` | flag | off | Region-crop the VAE ops (speed) |
| `--scr_log` | flag | off | Store per-step diagnostics |

These existing flags still matter: `--native_ctx relay`, `--relay_gain`, `--relay_text_bias`, `--relay_cut` (only with `brick`), `--hr_shift`, `--area_scale` (with `crpa`).

---

## 5. Diagnostics (logged with `--scr_log`)

Per stage-2 step j:
- `cons_j = ‖m_L ⊙ (x̂_L − f_L)‖ / ‖m_L ⊙ x̂_L‖`: how much the LR model rewrites the fake. A floor that does not decay means a systematic domain bias (W2). Expect the floor to fall in the order bilinear/haar > taef1 ≈ vae.
- `relay_j = ‖δ_H‖ / ‖m_H ⊙ x̂_H‖`.
- `t_ms_j`: wall time of the HR pass, LR pass and XScale calls (`torch.cuda.synchronize`).

Once per image, from the raw decodes before feathering:
- `E_pix`: RMSE between `area_down(dec_hr)` and `dec_lr` (512) on core pixels eroded by 8 px, divided by the std of `dec_lr` there. This is the pixel-space consistency the final blend relies on.
- `aniso`: mean over 32-px core cells of Ex/Ey, where Ex = mean (∂x I)² and Ey = mean (∂y I)² of `dec_hr` luminance. Also report the same quantity on a `--mask_pattern full` (uniform-HR) reference run with the same seed. Streaks give values ≫ the reference (the current output reaches 12–25).
- Optional: save `dec_hr` and `dec_lr` separately (`--scr_dump_streams`) so the HR stream can be inspected without the feather.

---

## 6. Unit and regression tests

- **T1 (Haar round trip):** `haar.down(haar.up(x)) == x` (max err < 1e-6, fp32).
- **T2 (exact replacement):** with `haar`, B = none, g = 1 and an LR-cell-aligned mask: after the relay, `down(x̂_H')` equals x̂_L on the region and `(I − UD)(x̂_H' − x̂_H) == 0` (verified offline to 4e-16).
- **T3 (promotion whiteness):** for i.i.d. ε̂_L and ξ, `ε_H` has per-2×2-block covariance ≈ I (off-diagonals < 0.02 at 1e5 samples).
- **T4 (legacy reproduction):** `XScale("bilinear")` equals `_resize_x0_to_x0(..., "bilinear")` / `(..., "avgpool")` bit-for-bit. A full pipeline run with all SCR flags at default reproduces an old `--native_ctx relay` output (same seed; identical image, or PSNR > 50 dB if GPU kernels are non-deterministic).
- **T5 (pixel-mode round trip, GPU):** report `‖down(up(x)) − x‖ / ‖x‖` and the shift (use `resizer_align.shift`) for `taef1` and `vae` on 8 real latents. This is not expected to be exact; record the numbers.
- **T6 (gain-0 regression):** `--relay_gain 0` must give the same HR result as a two-stream run with the relay disabled.

---

## 7. Experiment plan (in order; each gates the next)

Common options:

```bash
O="--prompts comm --cats face_bisected periodic_bisected --group_mode mean --only noer --native_ctx relay --scr_log --out_dir ./scr_out"
M1="--mask_pattern salblock --strip_w 2 --mask_hr_ratio 0.1"
M2="--mask_pattern bisecting --mask_hr_ratio 0.3"
M3="--mask_pattern scattered --strip_w 2 --mask_hr_ratio 0.3"
```

**E0. Source of the streaks** (no new code needed)

```bash
for M in "$M1" "$M2"; do for G in 0 1; do for TB in 0 auto; do
  python -m flux.run_seam --device cuda:0 $O $M --relay_resampler bilinear --relay_gain $G --relay_text_bias $TB \
         --tag_prefix e0_g${G}_tb${TB}_
done; done; done
```

- Prediction: streaks appear for **both** g = 0 and g = 1, so they come from the context-free HR pass (W1). `auto` text bias reduces them but does not remove them. P2 disagreement is worse at g = 0.
- If g = 0 has *no* streaks, the relay itself is the cause. Then check W3 (`--relay_cut 0.5`, or `--scr_band gauss` once implemented).

**E1. Operator measurement** (no new code; the tool exists)
- Run `python -m flux.resizer_align --device cuda:0` and record the full table (r_all, r_lo, r_hi, shift, ms) for up = repeat / bilinear / bicubic / pix_taef1 / pix_vae / lt_up / bi_up and down = box / pix_taef1 / pix_vae / lt_down.
- Prediction: r_lo(pix_*) > r_lo(box/bilinear), with |shift| < 1 px for the pix modes. If pix_taef1 is within ~0.02 of pix_vae, use `taef1` as the default.

**E2. Context** (first new code: `--scr_ctx crpa`)
- Run E0's best text-bias setting plus `--scr_ctx crpa`, still with bilinear and brick, on M1, M2, M3.
- Prediction: Ex/Ey drops to the reference level (streaks gone), and ghost eyes stay fixed. Residual: slight blur or ripples from W3/W4.

**E3. Operator and band**
- With `--scr_ctx crpa`, run `--scr_xscale {haar, taef1, vae}` × `--scr_band none`, plus `taef1 --scr_band gauss`.
- Predictions:
  - the `cons` floor falls in the order haar > taef1 ≈ vae;
  - `E_pix` is lowest for the pix modes;
  - `haar` may show 16-px block structure, because U_nn of a biased Δ is piecewise constant.

**E4. Final fill**
- Take the best E3 config and add `--scr_final_fill xscale`.
- Prediction: removes residual colour or texture seams within ~28 px of the core edge (W5).

**E5. Entry and schedule**
- Take the best config so far and add `--scr_lr_restart off --scr_promote preserve`, then also `--hr_shift 3.7552211585`.
- Prediction: brick-period mismatch for periods ≥ ~32 px is reduced. That band is pinned by both the relay and the noise realization. Finer brick texture may still differ (HR octave).
- Measure the brick period (autocorrelation peak) on `dec_hr` core cells vs `dec_lr`.

**E6. Paper 2×2** (best operator)
- `--scr_ctx {none, crpa}` × `--relay_gain {0, 1}`, on M1, M2, M3.

| | relay off (g=0) | relay on (g=1) |
|---|---|---|
| ctx none | stripes + disagreement | stripes, layout fixed (current) |
| ctx crpa | plain-CRPA P2 (ghost eyes) | **clean** |

- If this table comes out as predicted, it is the core analysis figure: context and consistency are separable failures with separable fixes.

**E7. Cost and generality**
- Measure seconds per image and stage-2 tokens per step vs E&R (pad 2) across fragmentation (scattered strip_w 2/4, ratio 0.1/0.3). E&R's HR stream reaches 100% of the image on scattered w2.
- Measure the per-step cost of `taef1` vs `vae` ops.
- Run one second model, or at least FLUX at a different resolution pair, for the "any DiT/VAE, zero training" claim.

**Report back after each experiment:** the images, the `scr` block of `meta.json` (`cons_j`, `relay_j`, `E_pix`, `aniso`, timings), and the E1 table.

---

## 8. Do not do (lessons from earlier phases)

- No fractional RoPE edits, no per-frequency RoPE amplitude changes (sinc), no keys between slots.
- Do not judge seams on the feathered `mixed.png` within 28 px of an edge. Use the raw stream decodes (§5).
- Do not add the low band twice. Keep the delta form; never add up(x̂_L) directly.
- Do not combine velocities of the two schedulers. Operate in x̂₀ space, then propagate (1−σ').
- Keep the training-free claim clean: `learned` is a comparison row only.

## 9. Open questions to answer from the runs

1. Is W1 the whole story for the streaks (E0)?
2. How large is the W2 floor per operator (`cons_j`, E1/E3)? Is `taef1` close enough to `vae`?
3. Does `crpa` context reintroduce scale bias in the HR octave (mini-bricks)? Does `--area_scale hr` help there?
4. Does `preserve` promotion plus the matched shift move the brick-scale result (E5), or is fine-texture scale an irreducible HR-octave prior?
5. Cost of the pixel-space ops relative to a DiT step. Is `--scr_crop` needed?
