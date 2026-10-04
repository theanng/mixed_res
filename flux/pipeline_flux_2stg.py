# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy

from einops import rearrange
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.utils import logging

from .pipeline_flux_base import (
    calculate_shift, 
    retrieve_timesteps, 
    FluxPipelineOutput,
    PipelineImageInput,
    FluxPipeline as FluxPipeline_BASE,
)
from .utils.dct_func import patch_dct_energy_mask
from .utils.seg_func import refine_mask_w_seg
from .utils.helpers import transform_base_mask, feather_mask_from_bool, quick_benchmark_gpu, align_latent_mean_shift_flexible


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _patch_cos_dist(a, b, pm):
    """Mean cosine distance (1 - cos) between a,b over the patch tokens. a,b:[b,seq,c] pm:[b,seq,1].
    Same [0,2] scale as the attention out_chg, so content and output changes are comparable."""
    m = pm[..., 0] > 0.5
    return (1.0 - F.cosine_similarity(a[m], b[m], dim=-1)).mean().item()


class FluxPipeline(FluxPipeline_BASE):

    def _vae1_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h*hw_scale, self._w*hw_scale, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]     
        return image   

    def _tvae_decode(self, latents, hw_scale=1, need_unpack=True) -> torch.Tensor:
        if need_unpack:
            latents = self._unpack_latents(latents, self._h*hw_scale, self._w*hw_scale, self.vae_scale_factor)
        image = self.tvae.decode(latents, return_dict=False)[0]     
        return image   

    def _pack_lt_w_scale(self, latents, hw_scale) -> torch.Tensor:
        lt_h = 2 * (int(self._h*hw_scale) // (self.vae_scale_factor * 2))
        lt_w = 2 * (int(self._w*hw_scale) // (self.vae_scale_factor * 2))
        latents = self._pack_latents(latents, latents.shape[0], self._c_lt, lt_h, lt_w)
        return latents # [B, 4096, 64]

    def _resize_x0_to_x0(
        self,
        latents_x0: torch.Tensor,
        resize_scale: float,
        mode: str = "",
        ori_scale: int = 1,
        dec_image: bool = False,
        align_mean: bool = True,
    ) -> torch.Tensor:
        """to obtain fake new-res latents x0"""
        
        assert resize_scale in [0.5, 2.0]
        new_scale = resize_scale * ori_scale

        lt = self._unpack_latents(latents_x0, self._h*ori_scale, self._w*ori_scale, self.vae_scale_factor)
        if resize_scale == 2.0:
            assert mode in ["", "repeat", "bi"]
            if mode == "": 
                new_lt = self.lt_upscaler(lt)
            elif mode == "bi": 
                new_lt = self.bi_upscaler(lt)
            elif mode == "repeat": 
                new_lt = lt.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        elif resize_scale == 0.5:
            assert mode in ["", "repeat"]
            if mode == "":
                new_lt = self.lt_downscaler(lt)
            elif mode == "repeat":
                new_lt = lt[..., ::2, ::2]
        if align_mean:
            new_lt, _ = align_latent_mean_shift_flexible(lt, new_lt)
        latents_new_x0 = self._pack_lt_w_scale(new_lt, new_scale) 

        # quick_benchmark_gpu(self._tvae_decode, latents_x0, ori_scale, iters=50, warmup=20, name='tvae')
        image = self._tvae_decode(latents_x0, ori_scale) if dec_image else None

        return latents_new_x0, image

    def _add_noise_at_t(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep.reshape(1).to(x0.device, dtype=torch.float32)[0] * 1e-3
        return (sigma * noise.to(torch.float32) + (1.0 - sigma) * x0.to(torch.float32)).to(x0.dtype)

    def _synthesize_latents_from_x0(
        self,
        latents_fake_x0: torch.Tensor,
        latents_orig: Optional[torch.Tensor],
        latents_noise: torch.Tensor,
        mask_bool: Optional[torch.Tensor],
        target_timestep: torch.Tensor,
    ) -> torch.Tensor:
        latents_fake = self._add_noise_at_t(latents_fake_x0, latents_noise, target_timestep)

        if mask_bool is not None and latents_orig is not None:
            mask_latent = mask_bool.to(latents_orig.dtype)
            latents_fake = latents_fake * mask_latent + latents_orig * (1.0 - mask_latent)

        return latents_fake

    @torch.no_grad()
    def _lowpass_packed(self, x, hw_scale, pad=16):
        """Keep only what an LR grid can represent: ideal low-pass at the LR Nyquist (0.25 cyc/px of the
        HR latent), reflect-padded. Near-idempotent, so one refresh locks the layout (error ~0.2 vs ~1.1
        for pool+bilinear) while leaving the band's own high frequencies almost untouched."""
        lt = self._unpack_latents(x, self._h * hw_scale, self._w * hw_scale, self.vae_scale_factor).float()
        cut = 0.5 / hw_scale
        p = F.pad(lt, (pad,) * 4, mode="reflect"); Hh, Ww = p.shape[-2:]
        fy = torch.fft.fftfreq(Hh, device=p.device).abs()[:, None]
        fx = torch.fft.rfftfreq(Ww, device=p.device)[None, :]
        lp = torch.fft.irfft2(torch.fft.rfft2(p) * ((fy <= cut) & (fx <= cut)), s=(Hh, Ww))
        return self._pack_lt_w_scale(lp[..., pad:-pad, pad:-pad].to(x.dtype), hw_scale)

    @torch.no_grad()
    def _native_lf_refresh(self, lt_up_lr_x0, hr_new, hr_x0, lr_new, hr_noise, lr_noise,
                           hr_seq_mask, bd1_mask, bd2_mask, next_t_hr, next_t_lr, hr_scale):
        """Native-context proxy refresh that does NOT throw away the HR band's detail.

        Plain E&R replaces the band with re-noised up(LR x0) every step, so the band can never become
        sharp; with a small core the band outweighs it and the core copies the blur (flat patches).
        Here the band keeps its OWN high frequencies and only its low frequencies are locked to the
        LR layout, applied directly to the noisy latent: x += (1 - sigma_next) * (LP(up_lr) - LP(own x0)).
        Tokens the HR stream never processes are set to re-noised up(LR x0). The LR shadow of the core is
        downsampled from a CLEAN HR composite, so never-processed tokens can't leak into the resizer."""
        seq = hr_seq_mask.float()
        band = seq * bd2_mask.float()                                  # processed, outside the core
        hr_x0_clean = hr_x0.float() * seq + lt_up_lr_x0.float() * (1.0 - seq)
        d = (self._lowpass_packed(lt_up_lr_x0, hr_scale).float()
             - self._lowpass_packed(hr_x0_clean.to(hr_x0.dtype), hr_scale).float())
        sig = next_t_hr.reshape(1).to(torch.float32)[0] * 1e-3
        hr = hr_new.float() + (1.0 - sig) * d * band
        far = self._add_noise_at_t(lt_up_lr_x0, hr_noise, next_t_hr).float()
        hr = (hr * seq + far * (1.0 - seq)).to(hr_new.dtype)
        lt_down, _ = self._resize_x0_to_x0(hr_x0_clean.to(hr_x0.dtype), 1.0 / hr_scale, "", ori_scale=hr_scale)
        lr = self._synthesize_latents_from_x0(lt_down, lr_new, lr_noise, bd1_mask, next_t_lr)
        return hr, lr

    @torch.no_grad()
    def _trace_dump(self, j, latents_hr_x0, latents_lr_x0, actual_hr_mask, core_hr_mask_LR,
                    feather_radius, hr_scale):
        """Decode the MIXED x0 estimate at stage-2 step j using the SAME path as the final render
        (LR native decode + bicubic up, HR decode, feathered blend), so the frames are directly
        comparable to the check_starv output -- the drift can be watched developing (flux.trace)."""
        import os
        lt_hr_fill, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, "repeat", ori_scale=1)
        hr_full = latents_hr_x0 * actual_hr_mask + lt_hr_fill * (1.0 - actual_hr_mask)
        dec_hr = self._vae1_decode(hr_full, hw_scale=hr_scale, need_unpack=True).float()
        dec_lr = self._vae1_decode(latents_lr_x0, hw_scale=1, need_unpack=True).float()
        dec_lr_up = F.interpolate(dec_lr, scale_factor=hr_scale, mode="bicubic",
                                  align_corners=False).clamp_(-1.0, 1.0)
        core_pix = transform_base_mask(core_hr_mask_LR, hr_scale, patch_size=16)
        feather = feather_mask_from_bool(core_pix, radius=feather_radius)
        img = dec_lr_up * (1.0 - feather) + dec_hr * feather
        self.image_processor.postprocess(img, output_type="pil")[0].save(
            os.path.join(self._trace_dir, f"{self._trace_tag}_step{j:02d}.png"))

    @torch.no_grad()
    def _run_content_branch(self, latents_lr, latents_hr, latents_lr_x0, latents_hr_noise,
                            stage2_lr_noise, t_lr, t_hr, hr_scale, core_hr_mask_LR, timestep_lr,
                            timestep_hr, guidance, pooled, prompt_embeds, text_ids, ids_lr, ids_hr):
        """Branch-at-t content-swap test (flux.content_probe). From the shared step-t state, run
        THREE measurement forwards that all re-noise to the CURRENT step t:
          A : patch stays LR, original content.
          B : patch promoted to HR (upsample cleaned LR x0 via resizer, re-noise) -> changes BOTH
              representation AND content (resizer detail).
          A': patch stays LR but content perturbed by a MATCHED magnitude (isolates 'output moves
              because content moved', at the same representation) -> the falsifiability control.
        Representation effect = out_chg(A->B) minus out_chg(A->A') at matched content change.
        Neighbour-query attention (mass/out/q, near + far ring) captured per branch by the _CONTENT
        hook. Result stashed on self._content_result for the probe."""
        from . import transformer_flux as TF
        tf = self.transformer
        cy, cx, half = self._content_target
        b, _, ht, wt = core_hr_mask_LR.shape
        y0, y1 = max(0, cy - half), min(ht, cy + half + 1)
        x0, x1 = max(0, cx - half), min(wt, cx + half + 1)
        pm = torch.zeros_like(core_hr_mask_LR)
        pm[:, :, y0:y1, x0:x1] = True
        maskA = core_hr_mask_LR & (~pm)                       # patch is NOT core -> LR
        maskB = core_hr_mask_LR | pm                          # patch IS core    -> HR
        pm_hr_tok = rearrange(transform_base_mask(pm, hr_scale, patch_size=1),
                              "b c h w -> b (h w) c").to(latents_hr.dtype)   # [b,seq_hr,1]
        pm_lr_tok = rearrange(transform_base_mask(pm, 1, patch_size=1),
                              "b c h w -> b (h w) c").to(latents_lr.dtype)    # [b,seq_lr,1]

        # B: promote patch to HR content, re-noised to the CURRENT step t
        lt_up_x0, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, "bi", ori_scale=1)      # B content
        lt_up_x0_rep, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, "repeat", ori_scale=1)  # A content
        if getattr(self, "_content_noise_match", True):
            # noise-match: give B's HR patch the SAME noise realisation as A's LR patch (the LR
            # noise tiled to HR), so A->B differs ONLY in representation -- not in the noise seed.
            hr_noise, _ = self._resize_x0_to_x0(stage2_lr_noise, hr_scale, "repeat", ori_scale=1)
        else:
            hr_noise = latents_hr_noise                       # fresh HR noise (confounds A->B)
        latents_hr_prom = self._add_noise_at_t(lt_up_x0, hr_noise, t_hr)
        latents_hr_B = latents_hr * (1.0 - pm_hr_tok) + latents_hr_prom * pm_hr_tok

        # A': perturb the LR patch x0 by a magnitude matched to B's content change, keep LR.
        cc_B = _patch_cos_dist(lt_up_x0, lt_up_x0_rep, pm_hr_tok)                # target (cosine dist)
        eps = torch.randn_like(latents_lr_x0)
        pnorm = (latents_lr_x0 * pm_lr_tok).norm().clamp(min=1e-6)
        enorm = (eps * pm_lr_tok).norm().clamp(min=1e-6)
        k = float((2.0 * cc_B) ** 0.5) * (pnorm / enorm)                        # rel L2 ~ sqrt(2*cc)
        lr_x0_pert = latents_lr_x0 + k * eps * pm_lr_tok
        lt_up_x0_reppert, _ = self._resize_x0_to_x0(lr_x0_pert, hr_scale, "repeat", ori_scale=1)  # A' content
        latents_lr_Ap = latents_lr * (1.0 - pm_lr_tok) + \
            self._add_noise_at_t(lr_x0_pert, stage2_lr_noise, t_lr) * pm_lr_tok

        # patch box in HR-grid coordinate units (matches token_info coords_hr)
        r0, r1 = 2 * y0, 2 * y1 - 1
        c0, c1 = 2 * x0, 2 * x1 - 1
        cyh, cxh = (r0 + r1) / 2.0, (c0 + c1) / 2.0
        pbox = max(r1 - r0, c1 - c0) / 2.0
        ring_w = float(getattr(self, "_content_ring", 4.0))
        far0 = pbox + ring_w + 2.0

        def _fwd(hs_lr, hs_hr, mask_hr, keep):
            tf._mixed_mask_cache = None                       # masks differ across calls
            return tf.forward_mixed(
                hidden_states_lr=hs_lr, hidden_states_hr=hs_hr,
                mask_lr_tokens=mask_hr, mask_lr_keep_tokens=keep,
                timestep_lr=timestep_lr / 1000, timestep_hr=timestep_hr / 1000,
                guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, img_ids_lr=ids_lr, img_ids_hr=ids_hr,
                joint_attention_kwargs=self.joint_attention_kwargs)

        TF._CONTENT.update(on=True, patch=(cyh, cxh), pbox=pbox, ring=(pbox, pbox + ring_w),
                           far=(far0, far0 + 8.0), rec=[])
        recs = {}
        for name, (hs_lr, hs_hr, mask_hr) in {
            "A":  (latents_lr,    latents_hr,   maskA),
            "B":  (latents_lr,    latents_hr_B, maskB),
            "Ap": (latents_lr_Ap, latents_hr,   maskA),
        }.items():
            TF._CONTENT["rec"] = []
            _fwd(hs_lr, hs_hr, mask_hr, ~mask_hr)
            recs[name] = list(TF._CONTENT["rec"])
        TF._CONTENT["on"] = False
        tf._mixed_mask_cache = None                           # let the main loop rebuild its own

        self._content_result = {
            "A": recs["A"], "B": recs["B"], "Ap": recs["Ap"],
            "patch": dict(cy=cy, cx=cx, half=half, cyh=cyh, cxh=cxh, pbox=pbox),
            "lt_up_x0": lt_up_x0.detach().float().cpu(),          # B content (HR, resizer detail)
            "lt_up_x0_rep": lt_up_x0_rep.detach().float().cpu(),  # A content (HR, no detail)
            "lt_up_x0_reppert": lt_up_x0_reppert.detach().float().cpu(),  # A' content (HR)
            "pm_hr_tok": pm_hr_tok.detach().float().cpu(),
        }

    @torch.no_grad()
    def _flow_invert(self, x0, img_ids, sigma_target, n_steps, guidance,
                     pooled_projections, encoder_hidden_states, txt_ids):
        """Rectified-flow inversion of the HR grid: integrate the PF-ODE UP in sigma from the
        clean latent x0 (sigma=0) to sigma_target, using the HR model's own velocity. The result
        is a noised latent at sigma_target that is STRUCTURE-LOCKED to x0 -- denoising it returns
        toward x0 with coherent high-freq, instead of the divergent high-freq a random re-noise
        seed produces. This is the flow-matching analog of DDIM inversion (PixelRush's structural
        prior). Euler update mirrors the scheduler: x += (s_next - s_cur) * v, ascending sigma."""
        bsz = x0.shape[0]
        x = x0.to(torch.float32)
        sigmas = torch.linspace(0.0, float(sigma_target), n_steps + 1, device=x0.device)
        for i in range(n_steps):
            s_cur = sigmas[i]
            s_next = sigmas[i + 1]
            timestep = s_cur.expand(bsz).to(x0.dtype)          # model expects sigma in [0,1]
            with self.transformer.cache_context("cond"):
                v = self.transformer(
                    hidden_states=x.to(x0.dtype),
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
            x = x + (s_next - s_cur) * v.to(torch.float32)
        # Diagnostic: does the inversion actually inject noise, or does it collapse toward x0?
        # A proper noised latent at sigma_target has |x| ~ |ref| and effective-noise ~ |rand|.
        # If |inv| << |ref| or |eff_eps| << |rand_eps|, the inversion under-noised (the sigma=0
        # start / off-manifold blurry x0 problem) -> denoise rebuilds the blur -> HR invisible.
        with torch.no_grad():
            st = max(float(sigma_target), 1e-6)
            rand = torch.randn_like(x0.to(torch.float32))
            ref = (1.0 - st) * x0.to(torch.float32) + st * rand
            eff_eps = (x - (1.0 - st) * x0.to(torch.float32)) / st
            print(f"[invert] sigma={st:.3f} steps={n_steps}  |x0|={x0.to(torch.float32).norm():.0f}"
                  f"  |inv|={x.norm():.0f}  |ref_rand|={ref.norm():.0f}"
                  f"  |eff_eps|={eff_eps.norm():.0f}  |rand_eps|={rand.norm():.0f}")
        return x.to(x0.dtype)

    def _get_saliency(
        self,
        img_lr: torch.Tensor,
    ):
        # print(img_lr.shape, img_lr.min(), img_lr.max(), img_lr.dtype) 
        # torch.Size([1, 3, 512, 512]) tensor(-1.2500, device='cuda:0', dtype=torch.bfloat16) tensor(1.2969, device='cuda:0', dtype=torch.bfloat16) torch.bfloat16
        saliency_map, img_lr_uint8 = None, None
        
        img_lr_uint8 = ((img_lr.float() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        img_lr_uint8 = F.interpolate(img_lr_uint8, self.gaze_shape)
        centerbias_tensor = self.centerbias_tensor.clone()
        centerbias_tensor = centerbias_tensor.expand(img_lr_uint8.shape[0], -1, -1)

        log_density_prediction = self.gaze_model(img_lr_uint8, centerbias_tensor)
        density_preds = torch.exp(log_density_prediction)
        if density_preds.ndim == 3:
            density_preds = density_preds.unsqueeze(1)
        lt_h = self._h // (self.vae_scale_factor * 2)
        lt_w = self._w // (self.vae_scale_factor * 2)
        saliency_map = F.interpolate(density_preds, size=(lt_h, lt_w))

        return saliency_map, img_lr_uint8

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents_lr: Optional[torch.FloatTensor] = None,
        latents_hr: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Mixed-resolution options
        hr_ratio: float = 0.25,
        hr_scale: int = 2,
        pad1: int = 2,
        pad2: int = 2, 
        mid_up_mode1: str = "repeat",
        mid_up_mode2: str = "",
        stg1_true_steps: int = 6,
        stg2_true_steps: int = 9,
        use_dct_mid_up_split: bool = True,
        mask_type: str = "w_seg",
        feather_radius: int = 28,
        boundary_grad: int = 0,  # >0: graded (soft-ramp) boundary bands instead of a hard step
    ):
        if num_images_per_prompt != 1:
            raise ValueError("`num_images_per_prompt` must be 1 in this pipeline.")
        assert callback_on_step_end is None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._h, self._w = height, width

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        num_inference_steps = stg1_true_steps + stg2_true_steps
        self._num_timesteps = num_inference_steps

        use_seg = False # mask_type == "base"
        if mask_type == "w_seg":
            use_seg = True

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if do_true_cfg:
            raise ValueError("`true_cfg_scale` path is not implemented in this pipeline.")

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        self._c_lt = num_channels_latents
        bsz = batch_size * num_images_per_prompt

        latents_lr_noise, latent_lr_image_ids = self.prepare_latents(
            bsz,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents_lr,
        )
        latents_hr_noise, latent_hr_image_ids = self.prepare_latents(
            bsz,
            num_channels_latents,
            height * hr_scale,
            width * hr_scale,
            prompt_embeds.dtype,
            device,
            generator,
            latents_hr,
        )
        stage2_lr_noise = latents_lr_noise

        image_seq_len_stg1 = latents_hr_noise.shape[1]

        p_h, p_w = 2, 2
        ht = (height // self.vae_scale_factor) // p_h
        wt = (width // self.vae_scale_factor) // p_w
        if ht * wt != latents_lr_noise.shape[1]:
            raise ValueError("Invalid LR token shape for mask construction.")

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(bsz)
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                bsz,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            _ = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                bsz,
            )

        mu = calculate_shift(
            image_seq_len_stg1,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        scheduler_stg2_lr = self.scheduler
        scheduler_stg2_hr = self.scheduler_hr
        stg2_start_idx_lr = num_inference_steps - stg2_true_steps
        stg2_start_idx_hr = num_inference_steps - stg2_true_steps

        timesteps_lr, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        timesteps_stg2_hr_full, _ = retrieve_timesteps(
            scheduler_stg2_hr,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        timesteps_stg1 = timesteps_lr[:stg1_true_steps]
        timesteps_stg2_lr = timesteps_lr[stg2_start_idx_lr:]
        timesteps_stg2_hr = timesteps_stg2_hr_full[stg2_start_idx_hr:]
        if len(timesteps_stg2_lr) != stg2_true_steps or len(timesteps_stg2_hr) != stg2_true_steps:
            raise RuntimeError("Invalid stage-2 timestep slicing.")

        latents_lr = latents_lr_noise.clone()
        latents_hr = None
        latents_lr_x0 = None
        latents_hr_x0 = None
        saliency_map = None

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=stg1_true_steps + stg2_true_steps) as progress_bar:
            # Stage-1: LR denoise only
            for i, t in enumerate(timesteps_stg1):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep = t.expand(bsz).to(latents_lr_noise.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents_lr,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_lr_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                latents_lr, latents_lr_x0 = self.scheduler.step(noise_pred, t, latents_lr, return_dict=False)

                progress_bar.update()

            # Stage-1 artifacts: hr_x0 / saliency / masks
            latents_hr_x0, stage1_img_lr = self._resize_x0_to_x0(latents_lr_x0, hr_scale, mode=mid_up_mode1, ori_scale=1, dec_image=True)
            latents_hr_x0 = latents_hr_x0.to(latents_lr.dtype)

            saliency_map, stage1_img_lr_uint8 = self._get_saliency(stage1_img_lr)
            imp_flat = saliency_map.view(bsz, -1)
            target_k = int(hr_ratio * imp_flat.shape[1])
            _, imp_idx = torch.topk(imp_flat, target_k, dim=1, largest=True)
            imp_mask = torch.zeros_like(imp_flat, dtype=torch.bool)
            imp_mask.scatter_(1, imp_idx, True)
            core_hr_mask_LR = imp_mask.view(*saliency_map.shape) # core hr token areas, in LR latent

            # Hand-written mask override (Track A): replace the saliency mask with a fixed geometry
            # so the cut lands exactly through the structure and boundary density is a controlled
            # variable. Skips the seg refinement. `custom_mask_pattern` set on the pipe by run_seam.
            _custom = getattr(self, "custom_mask_pattern", None)
            if _custom:
                from .mask_patterns import make_mask, make_sal_mask, boundary_fraction
                ht_m, wt_m = core_hr_mask_LR.shape[-2], core_hr_mask_LR.shape[-1]
                if _custom.startswith("sal"):
                    # saliency-placed masks (salblock / salcut / salcheck / saleyes): see make_sal_mask
                    core_hr_mask_LR = make_sal_mask(
                        _custom, saliency_map, getattr(self, "custom_mask_hr_ratio", 0.1),
                        strip_w=getattr(self, "custom_mask_strip_w", None))[..., :ht_m, :wt_m].contiguous()
                else:
                    m = make_mask(_custom, ht_m, wt_m, getattr(self, "custom_mask_hr_ratio", 0.3),
                                  strip_w=getattr(self, "custom_mask_strip_w", None),
                                  device=core_hr_mask_LR.device)
                    core_hr_mask_LR = m.expand_as(core_hr_mask_LR).contiguous()
                use_seg = False
                print(f"custom mask '{_custom}' boundary_frac={boundary_fraction(core_hr_mask_LR):.2f}")

            if use_seg:
                # 16 ms on H100
                core_hr_mask_LR_w_seg = refine_mask_w_seg(
                    self.seg_model, stage1_img_lr_uint8, core_hr_mask_LR, target_k*1.1, conf=0.05, pool_threshold=0.05)
                
                tmp_mask_lr = (~core_hr_mask_LR_w_seg).float()
                if pad1 > 0:
                    tmp_mask_lr = F.max_pool2d(tmp_mask_lr, kernel_size=pad1 * 2 + 1, stride=1, padding=pad1) 
                tmp_mask_hr = core_hr_mask_LR_w_seg.float()
                if pad2 > 0:
                    tmp_mask_hr = F.max_pool2d(tmp_mask_hr, kernel_size=pad2 * 2 + 1, stride=1, padding=pad2)
                if tmp_mask_lr.mean() < 0.95 and tmp_mask_hr.mean() < 0.95:
                    core_hr_mask_LR = core_hr_mask_LR_w_seg
                    print("use seg mask")
            
            print(f"mask ratio: {core_hr_mask_LR.float().mean():.2f}")
            self._last_hr_mask_LR = core_hr_mask_LR.detach().clone()  # [b,1,ht,wt] bool, for diagnosis

            # Native-context canvases: no attention call mixes resolutions. The LR stream sees a COMPLETE
            # LR image (real LR outside the core + proxy LR = down(HR x0) inside it); the HR stream sees the
            # core + a band of proxy HR = up(LR x0). Proxies are rebuilt every step by the E&R resizers and
            # never kept as output, so this reuses the E&R refresh path (forced on, band = native_band).
            _nc = getattr(self, "native_ctx", "off")
            if _nc != "off":
                pad1 = max(int(pad1), 1)
                pad2 = max(int(getattr(self, "native_band", 2)), 1)

            actual_lr_mask_bchw = ~core_hr_mask_LR
            if pad1 > 0:
                actual_lr_mask_bchw = (
                    F.max_pool2d(
                        actual_lr_mask_bchw.float(),
                        kernel_size=pad1 * 2 + 1,
                        stride=1,
                        padding=pad1,
                    ).to(torch.bool)
                )
                bd1_mask_bchw = core_hr_mask_LR
                bd1_mask = transform_base_mask(bd1_mask_bchw, 1, patch_size=1).to(latents_lr.dtype)
                if boundary_grad > 0:
                    # graded LR boundary: ramp native-LR (background, ->0) <-> downsampled-HR (core, ->1)
                    k = boundary_grad * 2 + 1
                    bd1_mask = F.avg_pool2d(bd1_mask.float(), kernel_size=k, stride=1,
                                            padding=boundary_grad).to(latents_lr.dtype)
                bd1_mask = rearrange(bd1_mask, "b c h w -> b (h w) c")
            if _nc != "off":
                actual_lr_mask_bchw = torch.ones_like(actual_lr_mask_bchw, dtype=torch.bool)   # full LR canvas

            actual_hr_mask_bchw_LR = copy.deepcopy(core_hr_mask_LR)
            if pad2 > 0:
                actual_hr_mask_bchw_LR = (
                    F.max_pool2d(
                        core_hr_mask_LR.float(),
                        kernel_size=pad2 * 2 + 1,
                        stride=1,
                        padding=pad2,
                    ).to(torch.bool)
                )
                bd2_mask_bchw_LR = ~core_hr_mask_LR # 不属于core
                bd2_mask = transform_base_mask(bd2_mask_bchw_LR, hr_scale, patch_size=1).to(latents_hr_x0.dtype)
                if boundary_grad > 0:
                    # graded HR boundary: ramp native-HR (core, bd2->0) <-> upsampled-LR (background, bd2->1).
                    # avg_pool of the hard 0/1 mask turns the step at the core edge into a linear ramp, so
                    # _synthesize blends sharp-HR-x0 and blurry-LR-x0 across the band => graded freq ceiling.
                    k = boundary_grad * 2 + 1
                    bd2_mask = F.avg_pool2d(bd2_mask.float(), kernel_size=k, stride=1,
                                            padding=boundary_grad).to(latents_hr_x0.dtype)
                bd2_mask = rearrange(bd2_mask, "b c h w -> b (h w) c")
            if pad2 > 0:
                print(f"[{'native' if _nc != 'off' else 'E&R'}] HR stream covers {actual_hr_mask_bchw_LR.float().mean():.2f} of the image "
                      f"(HR region {core_hr_mask_LR.float().mean():.2f} + band {pad2}); ~1.00 = full-HR run")

            # (1, 1, 32, 32) -> (1, 32, 32)   (1, 64, 64)
            actual_lr_mask = transform_base_mask(actual_lr_mask_bchw, 1, patch_size=1)
            actual_hr_mask = transform_base_mask(actual_hr_mask_bchw_LR, hr_scale, patch_size=1)
            # -> (1, 1024,1)   (1, 4096, 1)
            actual_lr_mask = rearrange(actual_lr_mask.to(latents_lr.dtype), "b c h w -> b (h w) c")
            actual_hr_mask = rearrange(actual_hr_mask.to(latents_lr.dtype), "b c h w -> b (h w) c")

            if use_dct_mid_up_split:
                dct_high_mask = patch_dct_energy_mask(
                    stage1_img_lr.float(),
                    threshold = 0.15,
                )[1]
                assert dct_high_mask.shape == core_hr_mask_LR.shape

                low_energy_core_mask_lr = core_hr_mask_LR & (~dct_high_mask)
                mode2_mask_hr = transform_base_mask(low_energy_core_mask_lr, hr_scale, patch_size=1)

                latents_hr_x0_mode2, _ = self._resize_x0_to_x0(latents_lr_x0, hr_scale, mode=mid_up_mode2, ori_scale=1, dec_image=False)
                latents_hr_x0_mode2 = latents_hr_x0_mode2.to(latents_lr.dtype)
                mode2_mask_hr = rearrange(mode2_mask_hr, "b c h w -> b (h w) c").to(latents_hr_x0.dtype)
                latents_hr_x0 = latents_hr_x0 * (1.0 - mode2_mask_hr) + latents_hr_x0_mode2 * mode2_mask_hr

            # Stage-2 init: re-noise from x0 at timestep index (stg2_tot_steps_* - stg2_true_steps)
            stage2_start_t_lr = timesteps_stg2_lr[0]
            stage2_start_t_hr = timesteps_stg2_hr[0]

            if getattr(self, "invert_hr_init", False):
                # STRUCTURE-LOCKED HR init: flow-invert the upsampled-LR x0 up to the start sigma,
                # instead of adding a fresh random HR noise. Keeps the HR grid phase-locked to LR
                # (less divergence -> smaller seam), the mechanism that should let us drop E&R.
                latents_hr = self._flow_invert(
                    latents_hr_x0, latent_hr_image_ids, float(stage2_start_t_hr) / 1000.0,
                    int(getattr(self, "invert_hr_steps", 4)), guidance,
                    pooled_prompt_embeds, prompt_embeds, text_ids)
            else:
                latents_hr = self._add_noise_at_t(latents_hr_x0, latents_hr_noise, stage2_start_t_hr)
            latents_lr = self._add_noise_at_t(latents_lr_x0, stage2_lr_noise, stage2_start_t_lr)

            if getattr(self, "_dump_hr_init", False):
                # DIAGNOSTIC (flux.size_check): how much fresh noise buries the LR structure at the
                # HR stage-2 start, and what the HR region looks like BEFORE it re-denoises. Compare
                # this decoded structure-init to the final image: if the eye drifts here->final, the
                # drift is generative (high-sigma re-noise), not a latent/packing misalignment.
                sig_hr = float(stage2_start_t_hr) / 1000.0
                print(f"[hr-init] stage-2 start sigma  HR={sig_hr:.3f}  LR={float(stage2_start_t_lr)/1000.0:.3f}"
                      f"   => HR init = {1-sig_hr:.0%} structure + {sig_hr:.0%} fresh noise")
                with torch.no_grad():
                    dec = self._vae1_decode(latents_hr_x0, hw_scale=hr_scale, need_unpack=True).float()
                    self.image_processor.postprocess(dec, output_type="pil")[0].save(
                        getattr(self, "_dump_hr_init_path", "hr_init_x0.png"))

            # Stage-2: mixed-resolution denoise
            scheduler_stg2_lr.set_begin_index(stg2_start_idx_lr)
            scheduler_stg2_hr.set_begin_index(stg2_start_idx_hr)
            for j, (t_lr, t_hr) in enumerate(zip(timesteps_stg2_lr, timesteps_stg2_hr)):
                if self.interrupt:
                    continue

                self.transformer._s2_step = j                              # stage-2 step (attn probe)
                self.transformer._s2_nsteps = len(timesteps_stg2_hr)
                self._current_timestep = t_lr
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep_lr = t_lr.expand(bsz).to(stage2_lr_noise.dtype)
                timestep_hr = t_hr.expand(bsz).to(stage2_lr_noise.dtype)
                next_t_lr = timesteps_stg2_lr[j + 1] if (j + 1) < len(timesteps_stg2_lr) else None
                next_t_hr = timesteps_stg2_hr[j + 1] if (j + 1) < len(timesteps_stg2_hr) else None

                with self.transformer.cache_context("cond"):
                    noise_pred_lr, noise_pred_hr = self.transformer.forward_mixed(
                        hidden_states_lr=latents_lr,
                        hidden_states_hr=latents_hr,
                        mask_lr_tokens=actual_hr_mask_bchw_LR,
                        mask_lr_keep_tokens=actual_lr_mask_bchw,
                        timestep_lr=timestep_lr / 1000,
                        timestep_hr=timestep_hr / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids_lr=latent_lr_image_ids,
                        img_ids_hr=latent_hr_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )

                latents_lr_new, latents_lr_x0_new = scheduler_stg2_lr.step(
                    noise_pred_lr,
                    t_lr,
                    latents_lr,
                    return_dict=False,
                )
                latents_hr_new, latents_hr_x0_new = scheduler_stg2_hr.step(
                    noise_pred_hr,
                    t_hr,
                    latents_hr,
                    return_dict=False,
                )

                # Content-swap branch-at-t (flux.content_probe): from the shared step-t state
                # (latents_lr/latents_hr are still the step-t INPUTS here), run the A/B/A' patch
                # measurement. Re-noises to the CURRENT step t (t_lr,t_hr), matching this forward.
                if (getattr(self, "_content_target", None) is not None
                        and j == getattr(self, "_content_step", 0)):
                    self._run_content_branch(
                        latents_lr=latents_lr, latents_hr=latents_hr, latents_lr_x0=latents_lr_x0_new,
                        latents_hr_noise=latents_hr_noise, stage2_lr_noise=stage2_lr_noise,
                        t_lr=t_lr, t_hr=t_hr, hr_scale=hr_scale,
                        core_hr_mask_LR=actual_hr_mask_bchw_LR, timestep_lr=timestep_lr,
                        timestep_hr=timestep_hr, guidance=guidance, pooled=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds, text_ids=text_ids,
                        ids_lr=latent_lr_image_ids, ids_hr=latent_hr_image_ids)

                if getattr(self, "_trace_steps", False):       # per-step x0 frame (flux.trace)
                    self._trace_dump(j, latents_hr_x0_new, latents_lr_x0_new, actual_hr_mask,
                                     core_hr_mask_LR, feather_radius, hr_scale)

                if (pad1 > 0) and (pad2 > 0) and next_t_hr is not None:
                    lt_up_lr_x0, _ = self._resize_x0_to_x0(latents_lr_x0_new, hr_scale, "bi", ori_scale=1)
                    latents_hr = self._synthesize_latents_from_x0(
                        lt_up_lr_x0,
                        latents_hr_new,
                        latents_hr_noise,
                        bd2_mask,
                        next_t_hr,
                    )
                else:
                    latents_hr = latents_hr_new
                    latents_hr_x0 = latents_hr_x0_new

                if (pad1 > 0) and (pad2 > 0) and next_t_lr is not None:
                    lt_down_hr_x0, _ = self._resize_x0_to_x0(latents_hr_x0_new, 1.0 / hr_scale, "", ori_scale=hr_scale)
                    latents_lr = self._synthesize_latents_from_x0(
                        lt_down_hr_x0,
                        latents_lr_new,
                        stage2_lr_noise,
                        bd1_mask,
                        next_t_lr,
                    )
                else:
                    latents_lr = latents_lr_new
                    latents_lr_x0 = latents_lr_x0_new

                # native-context: redo the proxy refresh so the HR band keeps its detail (see _native_lf_refresh)
                if (_nc != "off" and getattr(self, "native_band_mode", "lf") == "lf"
                        and next_t_hr is not None and next_t_lr is not None):
                    latents_hr, latents_lr = self._native_lf_refresh(
                        lt_up_lr_x0, latents_hr_new, latents_hr_x0_new, latents_lr_new,
                        latents_hr_noise, stage2_lr_noise, actual_hr_mask, bd1_mask, bd2_mask,
                        next_t_hr, next_t_lr, hr_scale)

                progress_bar.update()

        self._current_timestep = None
        
        # clear mask cache in DiT
        self.transformer._mixed_mask_cache = None

        # fill: lr with hr | hr with lr
        lt_lr_fake, _ = self._resize_x0_to_x0(latents_hr, 1./hr_scale, "repeat", ori_scale=hr_scale, align_mean=False)
        latents_lr = latents_lr * actual_lr_mask + lt_lr_fake * (1. - actual_lr_mask)
        lt_hr_fake, _ = self._resize_x0_to_x0(latents_lr, hr_scale, "repeat", ori_scale=1, align_mean=False)
        latents_hr = latents_hr * actual_hr_mask + lt_hr_fake * (1. - actual_hr_mask)

        # apply distance feather on mask
        dec_lr = self._vae1_decode(latents_lr, hw_scale=1, need_unpack=True)
        dec_lr_up = F.interpolate(dec_lr, scale_factor=hr_scale, mode="bicubic", align_corners=False).clamp_(-1.0, 1.0) 
        dec_hr = self._vae1_decode(latents_hr, hw_scale=2, need_unpack=True) # (1, 3, 1024, 1024)
        core_hr_mask_pix = transform_base_mask(core_hr_mask_LR, hr_scale, patch_size=16) 
        core_hr_mask_pix_feather = feather_mask_from_bool(core_hr_mask_pix, radius=feather_radius)
        image = dec_lr_up.float() * (1. - core_hr_mask_pix_feather) + dec_hr.float() * core_hr_mask_pix_feather

        image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
