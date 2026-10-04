# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    # apply_rotary_emb,
    # get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from .transformer_flux_base import (
    FluxAttention as BaseFluxAttention,
    FluxAttnProcessor as BaseFluxAttnProcessor,
    FluxIPAdapterAttnProcessor,
    FluxSingleTransformerBlock as BaseFluxSingleTransformerBlock,
    FluxTransformerBlock as BaseFluxTransformerBlock,
    _get_qkv_projections,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    sequence_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    assert sequence_dim == 1
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out

def get_1d_rotary_pos_embed(
    dim: int,
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. 
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim)) / linear_factor
    )  # [D/2]
    return freqs

def _filter_by_idx(
    tensor: torch.Tensor,
    idx: Optional[torch.Tensor],
) -> torch.Tensor:
    B, K = idx.shape
    idx_expand = idx.view(B, K, *([1] * (tensor.dim() - 2))).expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, dim=1, index=idx_expand)


# Temporary diagnostic: throttled print of grouped-key norm vs real-key norm, showing the
# OLD (mean of post-RMSNorm keys, RMS<1) and NEW (norm_k of pre-norm mean, RMS~1) ratios
# side by side in a single run. Set "max": 0 to silence.
_GRP_NORM_DBG = {"n": 0, "max": 4}

# kappa(Delta) channel probe: capture post-norm, PRE-RoPE content q/k (flux.channel_probe).
_KAPPA = {"on": False, "q": [], "k": [], "max": 8}


# Attention-weight capture (for flux.attn_probe). When token_info["attn_capture"] is set, each
# mixed-attention branch records, per key-TYPE (text/LR/HR), the mean softmax mass, the per-key
# mass, and the mean pre-softmax logit -- aggregated over a query subsample and heads. This is
# how we test "do HR queries attend to their own kind more than to LR" without dumping a huge
# token x token matrix. Cleared/read by the probe via flux.transformer_flux._ATTN_LOG.
_ATTN_LOG = []
_ATTN_MAX = 400            # throttle: stop capturing after this many branch-records
_ATTN_NQ = 128             # query subsample per capture


def _capture_attn(q, k, splits, branch, s2_step=-1):
    """q:[b,Lq,H,D] roped query, k:[b,Lk,H,D] roped key, splits:[(s,e,label)...] over key dim."""
    import math
    if len(_ATTN_LOG) >= _ATTN_MAX:
        return
    b, Lq, H, D = q.shape
    idx = torch.randperm(Lq, device=q.device)[:min(Lq, _ATTN_NQ)]
    qh = q[:, idx].permute(0, 2, 1, 3).float()                 # [b,H,nq,D]
    kh = k.permute(0, 2, 1, 3).float()                         # [b,H,Lk,D]
    logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(D)        # [b,H,nq,Lk]
    attn = logits.softmax(-1)
    # concentration: attention entropy per query (normalized to [0,1]; 1 = uniform = least
    # concentrated). Tests the 'mixed context / count imbalance displaces concentration' idea.
    ent = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(-1)   # [b,H,nq]
    rec = {"branch": branch, "step": int(s2_step),
           "entropy_norm": (ent / math.log(attn.shape[-1])).mean().item()}
    for s, e, lab in splits:
        if e <= s:
            continue
        rec[f"{lab}_mass"] = attn[..., s:e].sum(-1).mean().item()   # total mass on this type
        rec[f"{lab}_perkey"] = attn[..., s:e].mean().item()         # mass per key (count-normalized)
        rec[f"{lab}_logit"] = logits[..., s:e].mean().item()        # pre-softmax magnitude
        rec[f"{lab}_n"] = int(e - s)
        if e - s > 1:                                               # concentration WITHIN this type
            p = attn[..., s:e]                                      # e.g. HRq's HR block = HR->HR
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            se = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
            rec[f"{lab}_selfent"] = (se / math.log(e - s)).mean().item()   # 0..1, 1=uniform within type
    _ATTN_LOG.append(rec)


# Distance-profile capture: attention mass vs PHYSICAL distance (HR-grid units), accumulated per
# branch. Tests the reach asymmetry -- prediction: the HR-query profile decays ~2x faster in
# physical space than the LR-query profile (CRPA spends kappa's token budget in the fine stride).
_DIST_LOG = {}                          # branch -> [mass_hist, count_hist]  (torch, on cpu)
_DIST_BIN_W = 3.0                       # HR-grid units per bin
_DIST_NBINS = 32
_DIST_CNT = {"n": 0, "max": 400}


def _capture_dist(q, k, n_txt, q_coords, k_img_coords, branch, s2_step):
    """Per stage-2 STEP: attention mass/count vs physical distance. Keyed by 'branch@step'."""
    import math
    if _DIST_CNT["n"] >= _DIST_CNT["max"]:
        return
    _DIST_CNT["n"] += 1
    b, Lq, H, D = q.shape
    idx = torch.randperm(Lq, device=q.device)[:min(Lq, _ATTN_NQ)]
    qh = q[:, idx].permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    attn = ((qh @ kh.transpose(-1, -2)) / math.sqrt(D)).softmax(-1)      # [b,H,nq,Lk]
    attn_img = attn[..., n_txt:].mean(dim=(0, 1))                        # [nq, n_img]
    d = torch.cdist(q_coords[idx].float(), k_img_coords.float())        # [nq, n_img] physical
    binned = (d / _DIST_BIN_W).long().clamp(0, _DIST_NBINS - 1).reshape(-1)
    mass = torch.zeros(_DIST_NBINS, device=q.device).scatter_add_(0, binned, attn_img.reshape(-1))
    cnt = torch.zeros(_DIST_NBINS, device=q.device).scatter_add_(0, binned, torch.ones_like(attn_img.reshape(-1)))
    key = f"{branch}@{int(s2_step)}"
    if key not in _DIST_LOG:
        _DIST_LOG[key] = [mass.cpu(), cnt.cpu()]
    else:
        _DIST_LOG[key][0] += mass.cpu(); _DIST_LOG[key][1] += cnt.cpu()


# Spatial attention map, captured at the PROMOTE step (stage-2 step 0) and the LAST step, first
# layer of each (head-averaged). Tagged with step so the probe can compare promote vs last.
_SPATIAL = []
_SPATIAL_STEPS_DONE = set()
_SPATIAL_PER_STEP = 6


def _capture_spatial(q, k, n_txt, q_coords, k_img_coords, branch, n_hr_keys, s2_step, s2_nsteps):
    import math
    targets = {0, max(0, int(s2_nsteps) - 1)}
    if int(s2_step) not in targets or int(s2_step) in _SPATIAL_STEPS_DONE:
        return
    _SPATIAL_STEPS_DONE.add(int(s2_step))
    b, Lq, H, D = q.shape
    kh = k.permute(0, 2, 1, 3).float()
    for qi in torch.linspace(0, Lq - 1, _SPATIAL_PER_STEP).long().tolist():
        qv = q[:, qi:qi + 1].permute(0, 2, 1, 3).float()
        attn = ((qv @ kh.transpose(-1, -2)) / math.sqrt(D)).softmax(-1)
        w = attn[..., n_txt:].mean(dim=(0, 1)).squeeze(0)              # [n_img], head-averaged
        _SPATIAL.append({"q": q_coords[qi].cpu().numpy(), "kc": k_img_coords.cpu().numpy(),
                         "w": w.cpu().numpy(), "branch": branch, "n_hr": int(n_hr_keys),
                         "step": int(s2_step)})


# Content-swap capture (for flux.content_probe). The professor's hypothesis test: hold image
# content fixed, change only whether ONE patch is LR- or HR-represented, and ask whether the
# information arriving at NEIGHBOUR queries is faithful. Selection is by physical coordinate, so
# it works whether the patch is 1 lifted-LR key (condition A) or 4 HR keys (condition B): we sum
# the softmax mass landing anywhere inside the patch box, and record the attention OUTPUT vector
# for the ring of neighbour queries just outside it. Also a FAR ring (negative control: should be
# ~0 change at block 0) and the neighbour QUERY vector (drift control). Per-head; rds restriction
# and A/B/A' differencing happen in the probe. NOTE: condition B has 3 more key columns than A
# (1 LR key -> 4 HR keys), so the softmax normaliser differs very slightly (~1/2000) => mass_A vs
# mass_B is not exactly apples-to-apples; treat mass as indicative, out_chg as the primary signal.
_CONTENT = {"on": False, "patch": None, "pbox": 1.0, "ring": (1.0, 4.0), "far": (8.0, 16.0),
            "rec": [], "max_blocks": 8, "map": False, "map_block": 0}


def _capture_content(q, k, v, n_txt, q_coords, k_img_coords, branch, s2_step):
    """q,k,v:[b,L,H,D] (q,k roped, v raw). coords in HR-grid units. Records, per head: patch mass +
    attention-output + query vector, for a NEAR ring (just outside the patch) and a FAR ring."""
    import math
    C = _CONTENT
    if not C["on"] or len(C["rec"]) >= C["max_blocks"] or q_coords is None or k_img_coords is None:
        return
    cy, cx = C["patch"]; pb = C["pbox"]; r0, r1 = C["ring"]; f0, f1 = C["far"]
    blk = sum(1 for r in C["rec"] if r.get("branch") == branch)        # per-branch block index
    qc = q_coords.to(q.device).float(); kc = k_img_coords.to(q.device).float()
    cheb_q = torch.maximum((qc[:, 0] - cy).abs(), (qc[:, 1] - cx).abs())
    cheb_k = torch.maximum((kc[:, 0] - cy).abs(), (kc[:, 1] - cx).abs())
    near = torch.nonzero((cheb_q > r0) & (cheb_q <= r1)).squeeze(-1)   # ring just outside patch
    far = torch.nonzero((cheb_q > f0) & (cheb_q <= f1)).squeeze(-1)    # negative control
    ksel = torch.nonzero(cheb_k <= pb).squeeze(-1)                     # keys inside patch box
    if near.numel() == 0 or ksel.numel() == 0:
        C["rec"].append({"branch": branch, "block": blk, "step": int(s2_step), "empty": True,
                         "nq": int(near.numel()), "nk": int(ksel.numel())})
        return
    b, Lq, H, D = q.shape
    kh = k.permute(0, 2, 1, 3).float()                               # [b,H,Lk,D]
    vh = v.permute(0, 2, 1, 3).float()
    patch_cols = ksel + n_txt                                         # image keys sit after text

    def measure(sel):
        qh = q[:, sel].permute(0, 2, 1, 3).float()                   # [b,H,ns,D]
        attn = ((qh @ kh.transpose(-1, -2)) / math.sqrt(D)).softmax(-1)
        patch_attn = attn[..., patch_cols]                          # [b,H,ns,npatch]
        mass = patch_attn.sum(-1)                                   # [b,H,ns] visibility of patch
        out = attn @ vh                                            # [b,H,ns,D] FULL output (diluted)
        vpatch = vh[:, :, patch_cols, :]                           # [b,H,npatch,D]
        contrib = torch.matmul(patch_attn, vpatch)                 # [b,H,ns,D] output from the patch
        msg = contrib / mass.clamp(min=1e-8).unsqueeze(-1)         # mass-normalised patch MESSAGE
        qv = q[:, sel].permute(0, 2, 1, 3)                          # [b,H,ns,D]
        return (mass.mean(dim=(0, 2)).cpu(), out.mean(dim=(0, 2)).cpu(),
                msg.mean(dim=(0, 2)).cpu(), qv.mean(dim=(0, 2)).float().cpu())

    m_near, o_near, msg_near, q_near = measure(near)
    rec = {"branch": branch, "block": blk, "step": int(s2_step),
           "mass": m_near, "out": o_near, "msg": msg_near, "q": q_near,
           "nq": int(near.numel()), "nk": int(ksel.numel())}
    if far.numel() > 0:
        _, o_far, _, q_far = measure(far)
        rec["out_far"] = o_far; rec["q_far"] = q_far; rec["nq_far"] = int(far.numel())
    if C.get("map") and blk == C.get("map_block", 0):
        # ONE representative neighbour (the near query closest to the patch): its attention over all
        # image keys (head-averaged) + coords, so the probe can plot how that single neighbour looks
        # at the promoted region, A vs B.
        qi = near[int(cheb_q[near].argmin())]
        qh = q[:, qi:qi + 1].permute(0, 2, 1, 3).float()                 # [b,H,1,D]
        attn = ((qh @ kh.transpose(-1, -2)) / math.sqrt(D)).softmax(-1)   # [b,H,1,Lk]
        rec["att1"] = attn[..., n_txt:].mean(dim=(0, 1, 2)).cpu()         # [n_img] head-avg, 1 neighbour
        rec["q1coord"] = qc[qi].cpu()                                    # that neighbour's HR coord
        rec["kcoords"] = k_img_coords.detach().cpu()                     # [n_img,2] HR coords
    C["rec"].append(rec)


class FluxAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key_prenorm = key                 # keep pre-RMSNorm key for pool-before-norm grouping
        key = attn.norm_k(key)

        if _KAPPA["on"] and len(_KAPPA["q"]) < _KAPPA["max"]:
            # post-norm, PRE-RoPE content q/k for the kappa(Delta) channel probe (flux.channel_probe)
            _KAPPA["q"].append(query.detach().float().cpu())     # [b, seq, heads, head_dim]
            _KAPPA["k"].append(key.detach().float().cpu())

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

        mixed_res_info = token_info
        if mixed_res_info is not None:
            n_lr = mixed_res_info["n_lr"]
            rotary_emb_txt = mixed_res_info.get("rotary_txt") # ([512, 128], [...])
            rotary_emb_lr = mixed_res_info.get("rotary_lr") # ([1863, 128], [...])
            rotary_emb_hr = mixed_res_info.get("rotary_query") # ([1863, 128], [...])
            lr_overlap_idx_rev = mixed_res_info.get("lr_overlap_idx_rev")
            hr_overlap_idx_rev = mixed_res_info.get("hr_overlap_idx_rev")
            has_overlap_lr = mixed_res_info.get("has_overlap_lr", False)
            has_overlap_hr = mixed_res_info.get("has_overlap_hr", False)
            lr_rope_downsample_hr_tokens = mixed_res_info.get("lr_rope_downsample_hr_tokens", False)
            lr_group_mode = mixed_res_info.get("lr_group_mode", "mean")       # "topleft" | "mean" (when grouping on)
            attn_capture = mixed_res_info.get("attn_capture", False)          # record per-type attention (probe)
            coords_hr = mixed_res_info.get("coords_hr", None)                 # [N,2] HR-grid coords (probe)
            s2_step = mixed_res_info.get("s2_step", -1)
            s2_nsteps = mixed_res_info.get("s2_nsteps", 0)
            lr_expand = mixed_res_info.get("lr_expand", False)               # HR->LR footprint expansion
            lr_expand_steps = mixed_res_info.get("lr_expand_steps", None)    # (cos_steps, sin_steps) [4,hd]
            lr_expand_masscal = mixed_res_info.get("lr_expand_masscal", False)
            area_scale = mixed_res_info.get("area_scale", "off")             # area-weighted softmax: +log(area) on key logits
            area_scale = {True: "both", False: "off", None: "off"}.get(area_scale, area_scale)
            area_lr = area_scale in ("lr", "both")      # LR-query branch (under grouping: image<->text rebalance)
            area_hr = area_scale in ("hr", "both")      # HR-query branch (LR anchor +log4: the bubble fix)
            # native-context canvases: 'near' = HR queries see HR (core + proxy band) + far LR via CRPA;
            # 'full' = no cross-resolution keys at all. Text queries read the complete native LR canvas.
            native_ctx = mixed_res_info.get("native_ctx", "off")
            if native_ctx != "off":
                area_lr = False     # the LR canvas is an ordinary full LR image: leave its softmax untouched

            n_txt = rotary_emb_txt[0].shape[0] # 512
            _AREA_LOG = float(np.log(4.0))     # LR cell covers 2x2=4 HR cells (hr_scale=2)

            # split q, k, v
            if query.shape[1] > rotary_emb_lr[0].shape[0]:
                assert attn.added_kv_proj_dim is None
                encoder_query, img_q = query[:, :n_txt], query[:, n_txt:]
                encoder_key,   img_k = key[:, :n_txt],   key[:, n_txt:]
                encoder_value, img_v = value[:, :n_txt], value[:, n_txt:]
                img_k_prenorm = key_prenorm[:, n_txt:]
            else:
                assert attn.added_kv_proj_dim is not None
                img_q, img_k, img_v = query, key, value
                img_k_prenorm = key_prenorm

            encoder_query = apply_rotary_emb(encoder_query, rotary_emb_txt)
            encoder_key = apply_rotary_emb(encoder_key, rotary_emb_txt)

            batch_size, seq_len, _, head_dim = img_q.shape # [B, 1863, 24, 128]

            has_hr_tokens = seq_len > n_lr

            attn_outputs: List[torch.Tensor] = []

            attn_output_txt, attn_output_lr, attn_output_hr = None, None, None
            if n_lr > 0:
                assert rotary_emb_lr is not None

                cos_lr_all, sin_lr_all = rotary_emb_lr
                cos_lr_all, sin_lr_all = cos_lr_all.unsqueeze(0), sin_lr_all.unsqueeze(0)
                assert cos_lr_all.shape[1] >= n_lr

                cos_lr = cos_lr_all[:, :n_lr, ...]
                sin_lr = sin_lr_all[:, :n_lr, ...]

                key_lr = img_k[:, :n_lr, ...]
                value_lr = img_v[:, :n_lr, ...]

                key_context = key_lr
                value_context = value_lr
                cos_context = cos_lr
                sin_context = sin_lr

                if has_hr_tokens:
                    key_hr = img_k[:, n_lr:, ...]
                    value_hr = img_v[:, n_lr:, ...]
                    cos_hr = cos_lr_all[:, n_lr:, ...]
                    sin_hr = sin_lr_all[:, n_lr:, ...]

                    if has_overlap_hr:
                        key_hr = _filter_by_idx(key_hr, hr_overlap_idx_rev)
                        value_hr = _filter_by_idx(value_hr, hr_overlap_idx_rev)
                        cos_hr = _filter_by_idx(cos_hr, hr_overlap_idx_rev)
                        sin_hr = _filter_by_idx(sin_hr, hr_overlap_idx_rev)

                    if lr_rope_downsample_hr_tokens and key_hr.shape[1] > 0:
                        if key_hr.shape[1] % 4 != 0:
                            raise ValueError("lr_rope_downsample_hr_tokens requires HR tokens in 2x2 groups.")

                        num_groups = key_hr.shape[1] // 4
                        value_hr_groups = value_hr.view(batch_size, num_groups, 4, attn.heads, -1)

                        extra_dims = cos_hr.shape[2:]
                        cos_hr_groups = cos_hr.view(batch_size, num_groups, 4, *extra_dims)
                        sin_hr_groups = sin_hr.view(batch_size, num_groups, 4, *extra_dims)

                        if lr_group_mode == "topleft":
                            # ORIGINAL repo: post-norm top-left subsample (decimation) of the 2x2
                            # group -> aliased, non-representative, but exactly the released behavior.
                            key_hr_groups = key_hr.view(batch_size, num_groups, 4, attn.heads, -1)
                            key_hr_down = key_hr_groups[:, :, 0, ...]
                            value_hr_down = value_hr_groups[:, :, 0, ...]
                        else:  # "mean"
                            # KEY: pool the PRE-RMSNorm key, then apply norm_k ONCE. Since W_k is
                            # linear, mean(W_k x) = W_k mean(x); letting norm_k process the average
                            # restores RMS=1 with a single weight application -> the pooled key lands
                            # on the real-key distribution (magnitude AND direction). Averaging AFTER
                            # norm_k (mean of unit-RMS vectors) gives RMS<1 -> shrunk logits, OOD.
                            key_hr_pre = img_k_prenorm[:, n_lr:, ...]
                            if has_overlap_hr:
                                key_hr_pre = _filter_by_idx(key_hr_pre, hr_overlap_idx_rev)
                            key_hr_pre_groups = key_hr_pre.view(batch_size, num_groups, 4, attn.heads, -1)
                            key_hr_down = attn.norm_k(key_hr_pre_groups.mean(dim=2))
                            if _GRP_NORM_DBG["n"] < _GRP_NORM_DBG["max"]:
                                with torch.no_grad():
                                    real = key_hr.norm(dim=-1).mean()
                                    old = key_hr.view(batch_size, num_groups, 4, attn.heads, -1).mean(
                                        dim=2).norm(dim=-1).mean()
                                    new = key_hr_down.norm(dim=-1).mean()
                                print(f"[grp-norm] real={real.item():.3f}  "
                                      f"old(post-norm mean)={(old / real).item():.3f}x  "
                                      f"new(pre-norm+norm_k)={(new / real).item():.3f}x")
                                _GRP_NORM_DBG["n"] += 1
                            # VALUE is never normed -> plain mean is the correct coarse content.
                            value_hr_down = value_hr_groups.mean(dim=2)
                        # POSITION: keep the group's top-left grid point. cos/sin must NOT be
                        # averaged -- that would denormalize the rotation (cos^2+sin^2 != 1) and
                        # break RoPE's orthogonality. One representative token at the grid point.
                        cos_hr_down = cos_hr_groups[:, :, 0, ...]
                        sin_hr_down = sin_hr_groups[:, :, 0, ...]

                        key_context = torch.cat([key_context, key_hr_down], dim=1)
                        value_context = torch.cat([value_context, value_hr_down], dim=1)
                        cos_context = torch.cat([cos_context, cos_hr_down], dim=1)
                        sin_context = torch.cat([sin_context, sin_hr_down], dim=1)
                    else:
                        key_context = torch.cat([key_context, key_hr], dim=1)
                        value_context = torch.cat([value_context, value_hr], dim=1)
                        cos_context = torch.cat([cos_context, cos_hr], dim=1)
                        sin_context = torch.cat([sin_context, sin_hr], dim=1)

                query_lr = img_q[:, :n_lr, ...]
                query_lr = apply_rotary_emb(query_lr, (cos_lr.squeeze(0), sin_lr.squeeze(0)))
                key_context = apply_rotary_emb(key_context, (cos_context.squeeze(0), sin_context.squeeze(0)))

                # add text key, value
                key_context = torch.cat([encoder_key, key_context], dim=1)
                value_context = torch.cat([encoder_value, value_context], dim=1)

                if attn_capture and has_hr_tokens:
                    # LR query -> [txt | LR keys | HR keys]
                    n_txt_ = encoder_key.shape[1]
                    _capture_attn(query_lr, key_context,
                                  [(0, n_txt_, "txt"), (n_txt_, n_txt_ + n_lr, "LR"),
                                   (n_txt_ + n_lr, key_context.shape[1], "HR")], "LRq", s2_step)
                    # distance profile (clean coords only when no grouping / no overlap)
                    if (coords_hr is not None and not lr_rope_downsample_hr_tokens
                            and not has_overlap_lr and not has_overlap_hr
                            and s2_step in (0, max(0, s2_nsteps - 1))):
                        _capture_dist(query_lr, key_context, n_txt_, coords_hr[:n_lr],
                                      torch.cat([coords_hr[:n_lr], coords_hr[n_lr:]], dim=0), "LRq", s2_step)

                # content-swap probe: LR query view (needs ungrouped HR keys for clean coords)
                if (_CONTENT["on"] and has_hr_tokens and coords_hr is not None
                        and not lr_rope_downsample_hr_tokens and not has_overlap_lr and not has_overlap_hr):
                    _capture_content(query_lr, key_context, value_context, encoder_key.shape[1],
                                     coords_hr[:n_lr],
                                     torch.cat([coords_hr[:n_lr], coords_hr[n_lr:]], dim=0), "LRq", s2_step)

                # area-weighted softmax (LR query): +log(area) on image keys. keys=[txt|LR|HR].
                # LR keys cover area 4 -> +log4. HR keys: grouped ones represent 2x2=area 4 -> +log4;
                # ungrouped native-HR are area 1 -> +0. Text keys +0.
                lr_area_mask = None
                if area_lr:
                    n_txt_ = encoder_key.shape[1]
                    lr_area_mask = key_context.new_zeros((1, 1, 1, key_context.shape[1]))
                    lr_area_mask[..., n_txt_:n_txt_ + n_lr] += _AREA_LOG
                    if lr_rope_downsample_hr_tokens:
                        lr_area_mask[..., n_txt_ + n_lr:] += _AREA_LOG

                if (not has_hr_tokens) or native_ctx != "off":
                    # add text query (native mode: text reads the complete native LR canvas)
                    attn_output_txt_lr = dispatch_attention_fn(
                        torch.cat([encoder_query, query_lr], dim=1),
                        key_context, value_context,
                        attn_mask=None, backend=self._attention_backend,
                    )
                    attn_output_txt, attn_output_lr = attn_output_txt_lr[:, :encoder_query.shape[1]], attn_output_txt_lr[:, encoder_query.shape[1]:]
                else:
                    attn_output_lr = dispatch_attention_fn(
                        query_lr,
                        key_context, value_context,
                        attn_mask=lr_area_mask, backend=self._attention_backend,
                    )
                attn_outputs.append(attn_output_lr)

            if has_hr_tokens:
                assert rotary_emb_hr is not None

                cos_hr_full, sin_hr_full = rotary_emb_hr
                cos_hr_full, sin_hr_full = cos_hr_full.unsqueeze(0), sin_hr_full.unsqueeze(0)
                assert cos_hr_full.shape[1] >= seq_len

                cos_hr_tokens = cos_hr_full[:, n_lr:, ...]
                sin_hr_tokens = sin_hr_full[:, n_lr:, ...]

                query_hr = img_q[:, n_lr:, ...]
                key_hr_tokens = img_k[:, n_lr:, ...]
                value_hr_tokens = img_v[:, n_lr:, ...]

                key_hr_context = key_hr_tokens
                value_hr_context = value_hr_tokens
                cos_hr_context = cos_hr_tokens
                sin_hr_context = sin_hr_tokens

                query_hr = apply_rotary_emb(query_hr, (cos_hr_tokens.squeeze(0), sin_hr_tokens.squeeze(0)))
                hr_attn_mask = None

                if n_lr > 0:
                    key_lr_filtered = img_k[:, :n_lr, ...]
                    value_lr_filtered = img_v[:, :n_lr, ...]
                    cos_lr_filtered = cos_hr_full[:, :n_lr, ...]
                    sin_lr_filtered = sin_hr_full[:, :n_lr, ...]

                    if has_overlap_lr:
                        key_lr_filtered = _filter_by_idx(key_lr_filtered, lr_overlap_idx_rev)
                        value_lr_filtered = _filter_by_idx(value_lr_filtered, lr_overlap_idx_rev)
                        cos_lr_filtered = _filter_by_idx(cos_lr_filtered, lr_overlap_idx_rev)
                        sin_lr_filtered = _filter_by_idx(sin_lr_filtered, lr_overlap_idx_rev)

                    # rope the HR-token keys on their own (LR keys are handled next)
                    key_hr_context = apply_rotary_emb(key_hr_context, (cos_hr_context.squeeze(0), sin_hr_context.squeeze(0)))

                    if lr_expand and lr_expand_steps is not None:
                        # Variant A: replicate each (already-filtered) LR key to its 4 footprint
                        # HR-grid cells. Content is copied; the k-th copy's RoPE position = anchor
                        # (+) unit step, obtained by composing the anchor rotary with step-k's rotary
                        # (a pure rotation, so norm-safe). value is copied unchanged.
                        cos_a = cos_lr_filtered.squeeze(0)
                        sin_a = sin_lr_filtered.squeeze(0)
                        cos_steps, sin_steps = lr_expand_steps
                        K = cos_steps.shape[0]
                        lr_key_copies = []
                        for k in range(K):
                            cs = cos_steps[k].to(cos_a.dtype)
                            ss = sin_steps[k].to(sin_a.dtype)
                            cos_fp = cos_a * cs - sin_a * ss          # cos(theta_anchor + theta_step)
                            sin_fp = sin_a * cs + cos_a * ss          # sin(theta_anchor + theta_step)
                            lr_key_copies.append(apply_rotary_emb(key_lr_filtered, (cos_fp, sin_fp)))
                        key_lr_expanded = torch.cat(lr_key_copies, dim=1)          # copy-major
                        value_lr_expanded = value_lr_filtered.repeat(1, K, 1, 1)   # copy-major (matches)
                        key_hr_context = torch.cat([key_hr_context, key_lr_expanded], dim=1)
                        value_hr_context = torch.cat([value_hr_context, value_lr_expanded], dim=1)

                        if lr_expand_masscal:
                            # subtract log(K) from the replicated LR columns to cancel the K-fold
                            # softmax-mass inflation, isolating the envelope from the extra LR weight.
                            n_rep = key_lr_expanded.shape[1]
                            q_len = encoder_query.shape[1] + query_hr.shape[1]
                            k_len = encoder_key.shape[1] + key_hr_context.shape[1]
                            hr_attn_mask = key_hr_context.new_zeros((1, 1, q_len, k_len))
                            hr_attn_mask[..., -n_rep:] = -float(np.log(K))
                    elif native_ctx == "full":
                        pass    # no cross-resolution keys: HR queries see only HR (core + proxy band)
                    else:
                        # default: single-anchor LR key, roped with its anchor position
                        key_lr_ctx = apply_rotary_emb(key_lr_filtered, (cos_lr_filtered.squeeze(0), sin_lr_filtered.squeeze(0)))
                        key_hr_context = torch.cat([key_hr_context, key_lr_ctx], dim=1)
                        value_hr_context = torch.cat([value_hr_context, value_lr_filtered], dim=1)
                else:
                    key_hr_context = apply_rotary_emb(key_hr_context, (cos_hr_context.squeeze(0), sin_hr_context.squeeze(0)))

                # add text key, value
                key_hr_context = torch.cat([encoder_key, key_hr_context], dim=1)
                value_hr_context = torch.cat([encoder_value, value_hr_context], dim=1)

                if attn_capture and n_lr > 0:
                    # HR query -> [txt | HR keys | LR keys]  (the beard hypothesis lives here)
                    n_txt_ = encoder_key.shape[1]; n_hr_ = query_hr.shape[1]
                    _capture_attn(query_hr, key_hr_context,
                                  [(0, n_txt_, "txt"), (n_txt_, n_txt_ + n_hr_, "HR"),
                                   (n_txt_ + n_hr_, key_hr_context.shape[1], "LR")], "HRq", s2_step)
                    if (coords_hr is not None and not lr_rope_downsample_hr_tokens
                            and not has_overlap_lr and not has_overlap_hr
                            and s2_step in (0, max(0, s2_nsteps - 1))):
                        k_img_c = torch.cat([coords_hr[n_lr:], coords_hr[:n_lr]], dim=0)
                        _capture_dist(query_hr, key_hr_context, n_txt_, coords_hr[n_lr:], k_img_c, "HRq", s2_step)
                        _capture_spatial(query_hr, key_hr_context, n_txt_, coords_hr[n_lr:],
                                         k_img_c, "HRq", query_hr.shape[1], s2_step, s2_nsteps)

                # content-swap probe: HR query view (the LR-key lift -- the embedding under test).
                # Valid when LR keys are single-copy (no lr_expand) and no overlap re-indexing.
                if (_CONTENT["on"] and n_lr > 0 and coords_hr is not None
                        and not lr_expand and not has_overlap_lr and not has_overlap_hr):
                    _capture_content(query_hr, key_hr_context, value_hr_context, encoder_key.shape[1],
                                     coords_hr[n_lr:],
                                     torch.cat([coords_hr[n_lr:], coords_hr[:n_lr]], dim=0), "HRq", s2_step)

                # area-weighted softmax (HR query): +log(area) on image keys. keys=[txt|HR|LR].
                # HR keys area 1 -> +0; the lifted LR anchor keys cover area 4 -> +log4 (the pure,
                # content-preserving version of 'expand'). Skipped under lr_expand (own mass path).
                if area_hr and n_lr > 0 and not lr_expand:
                    n_txt_ = encoder_key.shape[1]; n_hr_ = query_hr.shape[1]
                    if hr_attn_mask is None:
                        q_len = encoder_query.shape[1] + query_hr.shape[1]
                        hr_attn_mask = key_hr_context.new_zeros((1, 1, q_len, key_hr_context.shape[1]))
                    hr_attn_mask[..., n_txt_ + n_hr_:] += _AREA_LOG     # LR anchor keys -> +log4

                if native_ctx != "off":
                    # text was already answered on the native LR canvas; HR queries only
                    _m = hr_attn_mask[..., encoder_query.shape[1]:, :] if hr_attn_mask is not None else None
                    attn_output_hr = dispatch_attention_fn(
                        query_hr, key_hr_context, value_hr_context,
                        attn_mask=_m, backend=self._attention_backend,
                    )
                else:
                    # add text query
                    attn_output_txt_hr = dispatch_attention_fn(
                        torch.cat([encoder_query, query_hr], dim=1),
                        key_hr_context, value_hr_context,
                        attn_mask=hr_attn_mask, backend=self._attention_backend,
                    )
                    attn_output_txt, attn_output_hr = attn_output_txt_hr[:, :encoder_query.shape[1]], attn_output_txt_hr[:, encoder_query.shape[1]:]
                attn_outputs.append(attn_output_hr)
            
            if attn_output_txt is None:
                print("[WARNING] no img tokens!")
                attn_output_txt = dispatch_attention_fn(
                    encoder_query, encoder_key, encoder_value, attn_mask=None, backend=self._attention_backend,
                )

            attn_outputs = [attn_output_txt,] + attn_outputs
            hidden_states = torch.cat(attn_outputs, dim=1)
            hidden_states = hidden_states.flatten(2, 3).type_as(query)

        else:
            if attn.added_kv_proj_dim is not None:
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttention(BaseFluxAttention):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        token_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_info is not None:
            kwargs = dict(kwargs)
            kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


@maybe_allow_in_graph
class FluxSingleTransformerBlock(BaseFluxSingleTransformerBlock):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio)
        processor = self.attn.processor
        if isinstance(processor, BaseFluxAttnProcessor):
            processor = FluxAttnProcessor()
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_info is not None:
            joint_attention_kwargs = dict(joint_attention_kwargs or {})
            joint_attention_kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


@maybe_allow_in_graph
class FluxTransformerBlock(BaseFluxTransformerBlock):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, qk_norm=qk_norm, eps=eps)
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        token_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_info is not None:
            joint_attention_kwargs = dict(joint_attention_kwargs or {})
            joint_attention_kwargs["token_info"] = token_info
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(
        self, theta: int, axes_dim: List[int],
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        freqs_dtype = torch.float64
        base_freqs = []
        for dim in self.axes_dim:
            _is_t = len(base_freqs) == 0 # t -> text
            base_freq = get_1d_rotary_pos_embed(
                dim,
                theta=self.theta,
                linear_factor=1 if _is_t else linear_factor,
                ntk_factor=1 if _is_t else ntk_factor**(dim / (dim - 2)),
                freqs_dtype=freqs_dtype,
            )
            base_freqs.append(base_freq)

        self.register_buffer("base_freqs_t", base_freqs[0], persistent=False)
        self.register_buffer("base_freqs_h", base_freqs[1], persistent=False)
        self.register_buffer("base_freqs_w", base_freqs[2], persistent=False)

    def get_rotary_for_coords(
        self,
        t_indices: torch.Tensor,
        h_indices: torch.Tensor,
        w_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns RoPE (cos, sin) embeddings for an arbitrary list of token coordinates.

        Args:
            t_indices: Tensor of shape [N] containing text indices in patch space (integer or float).
            h_indices: Tensor of shape [N] containing height indices in patch space (integer or float).
            w_indices: Tensor of shape [N] containing width indices in patch space (integer or float).

        Returns:
            Tuple of tensors (cos, sin), each of shape [N, attention_head_dim].
        """
        cos_t, sin_t = self._compute_axis_embeddings(t_indices, self.base_freqs_t)
        cos_h, sin_h = self._compute_axis_embeddings(h_indices, self.base_freqs_h)
        cos_w, sin_w = self._compute_axis_embeddings(w_indices, self.base_freqs_w)

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        return cos, sin
    
    def _compute_axis_embeddings(
        self, positions: torch.Tensor, base_freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine rotary factors for a single axis."""
        assert positions.dim() == 1

        positions = positions.to(dtype=base_freqs.dtype)
        base_freqs = base_freqs.to(device=positions.device)
        freqs = torch.outer(positions, base_freqs) # [S, D/2]
        cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() # [S, D]
        return cos, sin

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        assert ids.shape[-1] == 3 
        pos = ids.float()
        freqs_cos, freqs_sin = self.get_rotary_for_coords(pos[:, 0], pos[:, 1], pos[:, 2])
        return freqs_cos.to(ids.device), freqs_sin.to(ids.device)


def idx_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask:   (B, N), bool
    要求：每个 batch 内 True 的个数相同
    return: (B, K)
    """
    B, N = mask.shape
    K_per_batch = mask.sum(dim=1)

    # 断言每个 batch 的 K 一样（和你原来的 stack 前提一致）
    if not torch.all(K_per_batch == K_per_batch[0]):
        raise ValueError("各 batch 选中数量不同，无法在 dim=1 对齐。请先对齐或改用 padding。")

    K = int(K_per_batch[0].item())   # K may be 0: native-context mode filters out every token of a type

    # 计算每个 batch 的被选中列索引 (B, K)
    arange_n = torch.arange(N, device=mask.device)
    idx = arange_n.unsqueeze(0).expand(B, N)[mask].reshape(B, K)  # (B, K)
    return idx


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        seperate_attention: bool = True,
        lr_rope_downsample_hr_tokens: bool = True,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        base_use_rope_lr: bool = False,
    ):
        super().__init__()
        self.seperate_attention = seperate_attention
        self.lr_rope_downsample_hr_tokens = lr_rope_downsample_hr_tokens
        self.base_use_rope_lr = base_use_rope_lr
        if lr_rope_downsample_hr_tokens:
            assert seperate_attention
        if base_use_rope_lr: 
            assert not seperate_attention

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.pos_embed2 = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope, linear_factor=linear_factor, ntk_factor=ntk_factor)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self._mixed_mask_cache: Optional[Dict[str, Any]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _prepare_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: torch.Tensor,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        device = mask_lr_tokens.device
        batch_size = mask_lr_tokens.shape[0]

        mask_lr_keep_overlap = mask_lr_keep_tokens & mask_lr_tokens

        keep_lr = mask_lr_keep_tokens.reshape(batch_size, -1)

        y_idx = torch.arange(ht, device=device)
        x_idx = torch.arange(wt, device=device)
        Yg, Xg = torch.meshgrid(y_idx, x_idx, indexing="ij") # [32, 32] [32, 32]
        Yg_flat = Yg.reshape(-1)
        Xg_flat = Xg.reshape(-1)

        Ht2 = ht * 2
        Wt2 = wt * 2

        dy = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        dx = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)

        lr_linear_indices_all: List[torch.Tensor] = []
        hr_linear_indices_all: List[torch.Tensor] = []
        lr_overlap_token_flags: List[torch.Tensor] = []
        hr_overlap_token_flags: List[torch.Tensor] = []

        coords_h_lr_all: List[torch.Tensor] = []
        coords_w_lr_all: List[torch.Tensor] = []
        coords_h_hr_all: List[torch.Tensor] = []
        coords_w_hr_all: List[torch.Tensor] = []

        for bidx in range(batch_size):
            keep_lr_b = keep_lr[bidx]
            lr_idx = torch.nonzero(keep_lr_b, as_tuple=False).squeeze(1)
            lr_linear_indices_all.append(lr_idx)

            lr_overlap_flat = mask_lr_keep_overlap[bidx].reshape(-1)
            lr_overlap_flags = lr_overlap_flat.index_select(0, lr_idx).to(torch.bool)
            lr_overlap_token_flags.append(lr_overlap_flags)

            y_b = Yg_flat.index_select(0, lr_idx)
            x_b = Xg_flat.index_select(0, lr_idx)

            _sub = getattr(self, "lr_subcell", False)
            coords_h_lr = [y_b.float() if _sub else y_b]
            coords_w_lr = [x_b.float() if _sub else x_b]

            # center_lr: place every token at its footprint CENTROID for the RoPE coords, instead
            # of the top-left corner. HR cell is 1 HR-unit wide -> +0.5; LR cell is 2 -> +1. (Only
            # the coords used for rotary are shifted; the integer latent indices stay untouched.)
            _c = getattr(self, "center_lr", False)
            _lr_off = 1.0 if _c else 0.0
            _hr_off = 0.5 if _c else 0.0
            coords_h_hr = [y_b * 2 + _lr_off]
            coords_w_hr = [x_b * 2 + _lr_off]

            sel_hr_b = mask_lr_tokens[bidx]
            y_sel, x_sel = torch.nonzero(sel_hr_b, as_tuple=True)
            y_sel_numel = y_sel.numel()
            if y_sel_numel > 0:
                overlap_sel = mask_lr_keep_overlap[bidx][y_sel, x_sel]
                repeat_count = y_sel_numel

                y_hr_base = (2 * y_sel).repeat_interleave(4)
                x_hr_base = (2 * x_sel).repeat_interleave(4)
                y_hr_rep = y_hr_base + dy.repeat(repeat_count)
                x_hr_rep = x_hr_base + dx.repeat(repeat_count)

                hr_overlap_flags = overlap_sel.repeat_interleave(4).to(torch.bool)
                hr_overlap_token_flags.append(hr_overlap_flags)

                lin_hr = y_hr_rep * Wt2 + x_hr_rep
                hr_linear_indices_all.append(lin_hr)

                if getattr(self, "lr_subcell", False):
                    # lr_subcell: in the LR-query branch, put each HR sub-token at its own footprint
                    # centroid on the LR grid instead of stacking all 4 on the parent index. LR token
                    # y covers [y, y+1) and is indexed y (centroid y+0.5); sub-token dy covers
                    # [y+dy/2, y+dy/2+0.5), centroid y+dy/2+0.25 -> index y + (dy-0.5)/2 = y -/+ 0.25.
                    # Only LR-query x HR-key pairs get fractional offsets; LR-LR stays integer.
                    coords_h_lr.append(y_sel.repeat_interleave(4).float() + (dy.repeat(repeat_count).float() - 0.5) * 0.5)
                    coords_w_lr.append(x_sel.repeat_interleave(4).float() + (dx.repeat(repeat_count).float() - 0.5) * 0.5)
                else:
                    coords_h_lr.append(y_sel.repeat_interleave(4))
                    coords_w_lr.append(x_sel.repeat_interleave(4))

                coords_h_hr.append(y_hr_rep + _hr_off)
                coords_w_hr.append(x_hr_rep + _hr_off)
            else:
                empty_long = torch.empty(0, dtype=torch.long, device=device)
                hr_linear_indices_all.append(empty_long)
                hr_overlap_token_flags.append(torch.empty(0, dtype=torch.bool, device=device))

            coords_h_lr_all.append(torch.cat(coords_h_lr, dim=0))
            coords_w_lr_all.append(torch.cat(coords_w_lr, dim=0))

            coords_h_hr_all.append(torch.cat(coords_h_hr, dim=0))
            coords_w_hr_all.append(torch.cat(coords_w_hr, dim=0))

        seq_lens = [coords.numel() for coords in coords_h_lr_all]
        if len(set(seq_lens)) != 1:
            raise ValueError(f"forward_mixed currently requires equal token counts across batch. Got {seq_lens}.")
        N = seq_lens[0]

        lr_lengths = [idx.numel() for idx in lr_linear_indices_all]
        if len(set(lr_lengths)) != 1:
            raise ValueError("forward_mixed expects the same num LR tokens per batch element.")
        n_lr = lr_lengths[0]

        lr_overlap_mask_tensor = torch.stack(lr_overlap_token_flags, dim=0)
        hr_overlap_mask_tensor = torch.stack(hr_overlap_token_flags, dim=0)

        # see pipeline_flux._prepare_latent_image_ids
        coords_t_lr_b = torch.zeros_like(coords_h_lr_all[0])
        coords_t_hr_b = torch.zeros_like(coords_h_hr_all[0])
        bidx = 0
        rotary_emb_lr = self.pos_embed2.get_rotary_for_coords(
            coords_t_lr_b,
            coords_h_lr_all[bidx],
            coords_w_lr_all[bidx],
        )

        rotary_emb_hr_query = self.pos_embed2.get_rotary_for_coords(
            coords_t_hr_b,
            coords_h_hr_all[bidx],
            coords_w_hr_all[bidx],
        )

        has_overlap_hr = bool(hr_overlap_mask_tensor is not None and hr_overlap_mask_tensor.numel() > 0 and hr_overlap_mask_tensor.any())
        has_overlap_lr = bool(lr_overlap_mask_tensor is not None and lr_overlap_mask_tensor.numel() > 0 and lr_overlap_mask_tensor.any())
        lr_overlap_idx_rev = idx_from_mask(~lr_overlap_mask_tensor)
        hr_overlap_idx_rev = idx_from_mask(~hr_overlap_mask_tensor)

        return {
            "mask_lr_keep_tokens": mask_lr_keep_tokens,
            "mask_lr_keep_overlap": mask_lr_keep_overlap,
            "lr_indices": tuple(lr_linear_indices_all),
            "hr_indices": tuple(hr_linear_indices_all),
            "rotary_emb_lr": rotary_emb_lr,
            "rotary_emb_hr_query": rotary_emb_hr_query,
            "lr_overlap_idx_rev": lr_overlap_idx_rev,
            "hr_overlap_idx_rev": hr_overlap_idx_rev,
            "has_overlap_hr": has_overlap_hr,
            "has_overlap_lr": has_overlap_lr,
            "n_lr": n_lr,
            "N": N,
            # HR-grid physical coords of the mixed tokens [N,2] (LR anchors=2*LR, then HR),
            # for the attn distance-profile probe (reach in physical units).
            "coords_hr": torch.stack([coords_h_hr_all[0], coords_w_hr_all[0]], dim=-1),
        }

    def _get_mixed_mask_state(
        self,
        mask_lr_tokens: torch.Tensor,
        mask_lr_keep_tokens: Optional[torch.Tensor],
        batch_size: int,
        ht: int,
        wt: int,
    ) -> Dict[str, Any]:
        expected_shape = (batch_size, ht, wt)
        assert mask_lr_tokens.shape == expected_shape

        if mask_lr_keep_tokens is None:
            keep_mask = ~mask_lr_tokens
        else:
            assert mask_lr_keep_tokens.device == mask_lr_tokens.device
            assert mask_lr_keep_tokens.shape == mask_lr_tokens.shape
            assert mask_lr_keep_tokens.dtype == torch.bool
            keep_mask = mask_lr_keep_tokens

        cache = self._mixed_mask_cache
        if cache is not None:
            assert torch.equal(cache.get("mask_lr_tokens"), mask_lr_tokens)
            assert torch.equal(cache.get("mask_lr_keep_tokens"), keep_mask)
            return cache["data"]

        mask_clone = mask_lr_tokens.detach().clone()
        keep_clone = keep_mask.detach().clone()
        mask_state = self._prepare_mixed_mask_state(mask_clone, keep_mask, ht, wt)
        self._mixed_mask_cache = {
            "mask_lr_tokens": mask_clone,
            "mask_lr_keep_tokens": keep_clone,
            "data": mask_state,
        }
        return mask_state
        
    def forward_mixed(
        self,
        hidden_states_lr: torch.Tensor,
        hidden_states_hr: torch.Tensor,
        mask_lr_tokens: torch.Tensor = None,
        mask_lr_keep_tokens: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids_lr: torch.Tensor = None,
        img_ids_hr: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Mixed-resolution forward path.

        - Uses `mask_lr_tokens` (shape [B, 1, Ht, Wt] in patch space) to pick which HR sub-tokens (2x2 per selected
          LR token) to include from the HR latent. When `mask_lr_keep_tokens` is provided, it describes which LR tokens
          are kept for the LR branch, allowing overlap with the HR area.
        - Returns two noise predictions: for LR and HR latents respectively, both in 5D grids matching the inputs.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        b    = hidden_states_lr.shape[0] # [B, 1024, 64]
        b_hr = hidden_states_hr.shape[0] # [B, 4096, 64]
        b_m, _, ht, wt = mask_lr_tokens.shape # [B, 1, 32, 32] 
        assert b == b_hr and b == b_m 

        # Compute linear-embedded tokens for LR and HR
        tokens_lr_flat = self.x_embedder(hidden_states_lr) # [B, 1024, 3072]
        tokens_hr_flat = self.x_embedder(hidden_states_hr) # [B, 4096, 3072]

        # -----------------------------------------
        mask_state = self._get_mixed_mask_state(mask_lr_tokens.squeeze(1), mask_lr_keep_tokens.squeeze(1), b, ht, wt)

        lr_indices = mask_state["lr_indices"]
        hr_indices = mask_state["hr_indices"]
        N = mask_state["N"]
        n_lr = mask_state["n_lr"]

        mixed_tokens_list: List[torch.Tensor] = []
        for bidx in range(b):
            lr_tokens_b = tokens_lr_flat[bidx].index_select(0, lr_indices[bidx])
            hr_idx = hr_indices[bidx]
            if n_lr < N:
                hr_tokens_b = tokens_hr_flat[bidx].index_select(0, hr_idx)
                lr_tokens_b = torch.cat([lr_tokens_b, hr_tokens_b], dim=0)
            if lr_tokens_b.shape[0] != N:
                raise RuntimeError("Mixed token assembly produced inconsistent sequence length.")
            mixed_tokens_list.append(lr_tokens_b)

        mixed_tokens = torch.stack(mixed_tokens_list, dim=0) # [B, N, D]

        rotary_emb_lr = mask_state["rotary_emb_lr"]
        rotary_emb_hr_query = mask_state["rotary_emb_hr_query"]
        lr_overlap_idx_rev = mask_state["lr_overlap_idx_rev"]
        hr_overlap_idx_rev = mask_state["hr_overlap_idx_rev"]
        has_overlap_hr = mask_state["has_overlap_hr"]
        has_overlap_lr = mask_state["has_overlap_lr"]
        # ---- 

        timestep = timestep.to(hidden_states_lr.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states_lr.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        assert txt_ids.ndim == 2
        # ids = torch.cat((txt_ids, img_ids), dim=0) # [512,3] + [1024,3]
        # image_rotary_emb = self.pos_embed(ids) # ([1536, 128], [1536, 128])
        # rotary_emb_lr # ([1863, 128], [1863, 128])
        # rotary_emb_hr_query # ([1863, 128], [1863, 128])
        rotary_txt = self.pos_embed(txt_ids)

        assert 'ip_adapter_image_embeds' not in joint_attention_kwargs

        # ---------------------------------------
        # Transformer blocks over mixed tokens
        hidden_states = mixed_tokens  # [B, N, D]

        if n_lr == N or not self.seperate_attention:
            token_info = None
            if self.base_use_rope_lr:
                image_rotary_emb = (torch.cat([rotary_txt[0], rotary_emb_lr[0]]), torch.cat([rotary_txt[1], rotary_emb_lr[1]]))
            else:
                image_rotary_emb = (torch.cat([rotary_txt[0], rotary_emb_hr_query[0]]), torch.cat([rotary_txt[1], rotary_emb_hr_query[1]]))
        else:
            token_info = {
                "n_lr": n_lr,
                "rotary_txt": rotary_txt,
                "rotary_lr": rotary_emb_lr,
                "rotary_query": rotary_emb_hr_query,
                "lr_overlap_idx_rev": lr_overlap_idx_rev,
                "hr_overlap_idx_rev": hr_overlap_idx_rev,
                "has_overlap_lr": has_overlap_lr,
                "has_overlap_hr": has_overlap_hr,
                "lr_rope_downsample_hr_tokens": getattr(self.config, "lr_rope_downsample_hr_tokens", False),
            }
            image_rotary_emb = None

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                token_info=token_info,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                token_info=token_info,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states) # [B, 1863, 64]

        new_lr = hidden_states_lr.clone()
        new_hr = hidden_states_hr.clone()
        for bidx in range(b):
            new_lr[bidx].index_copy_(0, lr_indices[bidx], hidden_states[bidx, :n_lr])
            if n_lr < N:
                new_hr[bidx].index_copy_(0, hr_indices[bidx], hidden_states[bidx, n_lr:])

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return new_lr, new_hr
