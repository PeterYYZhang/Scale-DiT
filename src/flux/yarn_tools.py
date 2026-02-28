import torch
import numpy as np
import math
from typing import Union, Tuple, List, Optional
import torch.nn.functional as F
import torch.nn as nn
def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base)) #Inverse dim formula to find number of rotations


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """
    Find the correction range for NTK-by-parts interpolation.
    """
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim-1) #Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001 #Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    """
    Calculate the new base for NTK-aware scaling.
    """
    return base * (scale ** (dim / (dim - 2)))


def get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,
        yarn=True,
        max_pe_len=None,
        ori_max_pe_len=64,
        current_timestep=1.0,
        resonance: bool = True,
        resonance_min_rot: float = 1.0,
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    device = pos.device
    dtype  = freqs_dtype

    # Dimension indices (even positions) matching the original implementation.
    idx = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim  # α_i = i/D

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=dtype, device=device)
        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)     # s >= 1

        # =======================
        # === Resonance alignment ===
        # =======================                
        if resonance:
            L = torch.tensor(float(ori_max_pe_len), dtype=dtype, device=device)
            theta_t = torch.tensor(theta, dtype=dtype, device=device)
            inv_freq = torch.exp(-idx * torch.log(theta_t))
            rot = (L * inv_freq) / (2.0 * math.pi)  # r(α) = L * θ^{-α} / 2π
            use_resonance = rot >= 1.0
            rot_rounded = torch.round(rot)
            k = torch.clamp(rot_rounded, min=resonance_min_rot)
            target_inv_freq = (2.0 * math.pi * k) / L  # = θ^{-alpha_res}
            alpha_res = -torch.log(target_inv_freq) / torch.log(theta_t)
            idx_eff = torch.where(use_resonance, alpha_res, idx)
        else:
            idx_eff = idx


        # ---- YaRN basis spectra (using resonance-aligned idx_eff) ----
        beta_0, beta_1 = 1.25, 0.75
        gamma_0, gamma_1 = 16, 2

        # base: 1 / θ^{α̃}
        # freqs_base = 1.0 / torch.exp(idx_eff * torch.log(torch.tensor(theta, dtype=dtype, device=device)))  # [D/2]
        freqs_base = 1.0 / torch.exp(idx_eff * torch.log(torch.tensor(theta, dtype=dtype, device=device)))
        # linear(PI): base / s
        freqs_linear = freqs_base / scale

        # NTK-aware: use new_base^α̃
        new_base = find_newbase_ntk(dim, theta, scale)
        if isinstance(new_base, torch.Tensor) and new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        new_base = torch.tensor(float(new_base), dtype=dtype, device=device)
        freqs_ntk  = 1.0 / torch.exp(idx_eff * torch.log(new_base))  # [D/2]

        # ---- YaRN β band: linear ↔ NTK interpolation (same as original logic) ----
        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low); high = min(dim // 2, high)
        mask_beta = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(dtype))
        freqs = freqs_linear * (1 - mask_beta) + freqs_ntk * mask_beta

        # ---- YaRN γ band: high frequencies fall back to the base spectrum (same as original logic) ----
        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low); high = min(dim // 2, high)
        mask_gamma = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(dtype))
        freqs = freqs * (1 - mask_gamma) + freqs_base * mask_gamma

    else:
        # Within the training window or when YARN is disabled: keep the original RoPE (supports ntk_factor / linear_factor).
        theta_ntk = theta * ntk_factor
        idx0 = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
        freqs = 1.0 / torch.exp(idx0 * torch.log(torch.tensor(theta_ntk, dtype=dtype, device=device)))
        freqs = freqs / linear_factor

    # Phase outer product.
    freqs = torch.outer(pos if isinstance(pos, torch.Tensor) else torch.tensor(pos, device=device), freqs)

    if freqs.device.type == "npu":
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            mscale = torch.where(scale <= 1., torch.tensor(1.0, device=scale.device, dtype=scale.dtype),
                                 0.1 * torch.log(scale) + 1.0)
            freqs_cos = freqs_cos * mscale
            freqs_sin = freqs_sin * mscale
        return freqs_cos, freqs_sin
    elif use_real:
        return torch.cat([freqs.cos(), freqs.cos()], dim=-1).float(), torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


class FluxPosEmbed(nn.Module):
    def __init__(
            self,
            theta: int,
            axes_dim: List[int],
            method: str = 'yarn',
            base_resolution_hw: Optional[Tuple[int, int]] = (4096, 4096),  # Training (H, W) resolution in pixels used to decouple YARN.
            axis_order: str = "t-h-w",  # Axis semantics for ids; defaults to [time/text, H, W].
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method

        # Keep the original square-image defaults.
        self.base_resolution = 2048
        self.patch_size = 16
        self.base_patches = self.base_resolution // self.patch_size  # Fallback only.

        # Vision-YARN keeps independent training baseline patch counts for H and W.
        if base_resolution_hw is None:
            # Preserve legacy behavior where H_train == W_train == base_resolution.
            H_train_px, W_train_px = self.base_resolution, self.base_resolution
        else:
            H_train_px, W_train_px = base_resolution_hw

        self.base_patches_hw = (
            max(1, H_train_px // self.patch_size),  # H_train / ps
            max(1, W_train_px // self.patch_size),  # W_train / ps
        )

        # Specify the axis semantics for ids (change this if your ordering differs).
        axis_order = axis_order.lower().replace("_", "").replace("-", "")
        if axis_order not in ("thw", "twh"):
            raise ValueError("axis_order must be 't-h-w' or 't-w-h'")
        self.axis_order = axis_order

    @staticmethod
    def _span_length(pos_1d: torch.Tensor) -> int:
        # Robust span length = max - min + 1 to guard against offsets.
        return int(pos_1d.max().item() - pos_1d.min().item() + 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: [S, n_axes], typically cat(txt_ids, img_ids) with n_axes == len(axes_dim).
        For axis_order='t-h-w': i=0 -> text/time, i=1 -> H, i=2 -> W (use 't-w-h' if swapped).
        """
        n_axes = ids.shape[-1]
        assert n_axes == len(self.axes_dim), "axes_dim must match the number of axes in ids"

        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        # Determine the reference patch count per axis based on axis_order (only affects image H/W axes).
        # thw: i=1->H, i=2->W ;  twh: i=1->W, i=2->H
        if self.axis_order == "thw":
            axis_to_base_len = {0: None, 1: self.base_patches_hw[0], 2: self.base_patches_hw[1]}
        else:  # "twh"
            axis_to_base_len = {0: None, 1: self.base_patches_hw[1], 2: self.base_patches_hw[0]}

        for i in range(n_axes):
            common_kwargs = {
                'dim': self.axes_dim[i],
                'pos': pos[:, i],
                'theta': self.theta,
                'repeat_interleave_real': True,
                'use_real': True,
                'freqs_dtype': freqs_dtype,
            }

            # === Core: decide per axis whether to enable YARN using that axis's own training baseline length. ===
            base_len = axis_to_base_len.get(i, None)

            if self.method == 'yarn' and base_len is not None:
                current_len = self._span_length(pos[:, i])        # Current patch count for this axis (H_cur or W_cur).
                if current_len > base_len:
                    max_pe_len = torch.tensor(current_len, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=base_len,           # Each axis uses its own training baseline length (Vision-YARN key idea).
                    )
                else:
                    # Still within the training window, so YARN is not required.
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                # Non-image axes (such as t/text) or non-YARN methods stay unchanged.
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin