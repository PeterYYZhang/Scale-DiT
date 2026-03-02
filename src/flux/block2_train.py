# for training using 1K resolution images, we don't need acceleration since it is already fast enough, use FSDP for stability
import torch
import os
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict, Any, Callable, Tuple
from diffusers.models.attention_processor import Attention, F
from diffusers.models.attention_dispatch import dispatch_attention_fn
import sys
sys.path.append("..")
sys.path.append(".")
try:
    from .lora_controller import enable_lora
except:
    from lora_controller import enable_lora
# from .lora_controller import enable_lora
import math
from diffusers.models.embeddings import apply_rotary_emb
import random
import logging
import torch.nn as nn


# Sage / Sparge blocksparse attention (CUDA)
SAGE_CORE_AVAILABLE = False
try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda  # type: ignore
    SAGE_CORE_AVAILABLE = True
except Exception:
    SAGE_CORE_AVAILABLE = False

SAGE_BLOCKSPARSE_AVAILABLE = SAGE_CORE_AVAILABLE


def _call_sage_blocksparse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: Optional[torch.Tensor],
    model_config: Dict[str, Any],
) -> torch.Tensor:
    if not SAGE_CORE_AVAILABLE:
        raise RuntimeError("Sage blocksparse attention is not available")
    return block_sparse_sage2_attn_cuda(
        query,
        key,
        value,
        mask_id=block_map,
        dropout_p=0.0,
        scale=query.shape[-1] ** -0.5,
        smooth_k=bool(model_config.get("sage_smooth_k", True)),
        pvthreshd=int(model_config.get("sage_pvthreshd", 50)),
        attention_sink=bool(model_config.get("sage_attention_sink", False)),
        tensor_layout="HND",
        output_dtype=query.dtype,
        return_sparsity=False,
    )


# Cache Sage masks loaded from disk (CPU) by shape
_SAGE_BLOCK_MASK_CACHE: Dict[Tuple, torch.Tensor] = {}


def _load_sage_mask_thl_128x64(
    model_config: Dict[str, Any],
    *,
    batch_size: int,
    heads: int,
) -> torch.Tensor:
    """
    Load a precomputed THL mask on Sage's asymmetric grid:
      - Q-block = 128 tokens
      - K-block = 64 tokens

    Returns a CPU tensor (typically bool) that will be moved/cast on demand.
    """
    # Defaults (keep aligned with block2.py)
    model_config.setdefault("use_sage_blocksparse", True)
    model_config.setdefault("sage_kernel_blk_q", 128)
    model_config.setdefault("sage_kernel_blk_k", 64)
    model_config.setdefault("sage_smooth_k", True)
    model_config.setdefault("sage_pvthreshd", 50)
    model_config.setdefault("sage_attention_sink", False)

    img_h, img_w = model_config.get("image_size", (4096, 4096))
    cache_key = ("SAGE_THL_128x64_mask", int(batch_size), int(heads), int(img_h), int(img_w))
    cached = _SAGE_BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Precomputed mask file paths keyed by image size (H, W)
    base_dir = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks"
    size_to_path = {
        (6144, 6144): f"{base_dir}/(6144, 6144)x(6144, 6144)_downsampled_128_test-window.pt",
        (4096, 4096): f"{base_dir}/(4096, 4096)x(4096, 4096)_downsampled_128_test-window.pt",
        (2048, 6144): f"{base_dir}/(2048, 6144)x(2048, 6144)_downsampled_128_test-window.pt",
        (1024, 1024): f"{base_dir}/(1024, 1024)x(1024, 1024)_downsampled_128_test-window.pt",
        (2048, 2048): f"{base_dir}/(2048, 2048)x(2048, 2048)_downsampled_128_test-window.pt",
        (3072, 3072): f"{base_dir}/(3072, 3072)x(3072, 3072)_downsampled_128_test-window.pt",
        (8192, 8192): f"{base_dir}/(8192, 8192)x(8192, 8192)_downsampled_128_test-window.pt",
        (4096, 8192): f"{base_dir}/(4096, 8192)x(4096, 8192)_downsampled_128_test-window.pt",
        (6144, 4096): f"{base_dir}/(6144, 4096)x(6144, 4096)_downsampled_128_test-window.pt",
        (4096, 3072): f"{base_dir}/(4096, 3072)x(4096, 3072)_downsampled_128_test-window.pt",
        (8192, 6144): f"{base_dir}/(8192, 6144)x(8192, 6144)_downsampled_128_test-window.pt",
        (4096, 2048): f"{base_dir}/(4096, 2048)x(4096, 2048)_downsampled_128_test-window.pt",
        (8192, 4096): f"{base_dir}/(8192, 4096)x(8192, 4096)_downsampled_128_test-window.pt",
    }
    mask_path = size_to_path.get((int(img_h), int(img_w)))
    if mask_path is None:
        raise ValueError(f"Image size {img_h} not supported, please build the mask first.")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Block mask file does not exist: {mask_path}")

    block_mask = torch.load(mask_path, map_location="cpu")
    if isinstance(block_mask, torch.Tensor):
        block_mask = block_mask.to(torch.bool)

    # Expand to (B, H, Qb, Kb) if needed
    if block_mask.dim() == 2:
        block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, heads, -1, -1)
    elif block_mask.dim() == 3:
        block_mask = block_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif block_mask.dim() == 4:
        if block_mask.shape[0] != batch_size or block_mask.shape[1] != heads:
            block_mask = block_mask.expand(batch_size, heads, -1, -1)

    _SAGE_BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


# Optional: Check for Flash Attention support
FLASH_ATTENTION_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')

# Optional: Import BlockSparseAttnFun for even better performance
try:
    from block_sparse_attn import block_sparse_attn_func  # type: ignore
    BLOCK_SPARSE_ATTENTION_AVAILABLE = True
except Exception:
    BLOCK_SPARSE_ATTENTION_AVAILABLE = False


# Cache for block-sparse masks to avoid rebuilding every step
_BLOCK_MASK_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
# Cache for precomputed dense attention mask loaded from disk
_PRECOMPUTED_DENSE_MASK: Optional[torch.Tensor] = None


def _normalize_attn_mask_for_scores(
    attn_mask: Optional[torch.Tensor],
    q_len: int,
    k_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Make an attention mask broadcastable to attention scores of shape [B, H, q_len, k_len].

    This codebase sometimes loads/builds masks using config-derived sizes that can drift
    from the true token sequence lengths (e.g. different packing/tiling). When that
    happens, `masked_fill` will crash with a shape mismatch. We defensively crop/pad
    the last 2 dims to match (q_len, k_len).

    - For boolean masks: True == keep, False == mask out.
    - For additive masks: values are added to scores; padding uses 0.
    """
    if attn_mask is None:
        return None

    attn_mask = attn_mask.to(device, non_blocking=True)

    if attn_mask.ndim == 2:
        mq, mk = attn_mask.shape
        # Fast path
        if mq == q_len and mk == k_len:
            return attn_mask

        # Crop if too large
        cropped = attn_mask[: min(mq, q_len), : min(mk, k_len)]

        # Pad if too small
        pad_q = q_len - cropped.shape[0]
        pad_k = k_len - cropped.shape[1]
        if pad_q > 0 or pad_k > 0:
            if cropped.dtype == torch.bool:
                fill = True
            else:
                fill = 0
            out = torch.full((q_len, k_len), fill_value=fill, dtype=cropped.dtype, device=device)
            out[: cropped.shape[0], : cropped.shape[1]] = cropped
            return out

        return cropped

    if attn_mask.ndim == 4:
        # [B, H, Q, K] (or broadcastable variants)
        mq, mk = attn_mask.shape[-2], attn_mask.shape[-1]
        if mq == q_len and mk == k_len:
            return attn_mask

        cropped = attn_mask[..., : min(mq, q_len), : min(mk, k_len)]
        pad_q = q_len - cropped.shape[-2]
        pad_k = k_len - cropped.shape[-1]
        if pad_q > 0 or pad_k > 0:
            if cropped.dtype == torch.bool:
                fill = True
            else:
                fill = 0
            out_shape = (*cropped.shape[:-2], q_len, k_len)
            out = torch.full(out_shape, fill_value=fill, dtype=cropped.dtype, device=device)
            out[..., : cropped.shape[-2], : cropped.shape[-1]] = cropped
            return out

        return cropped

    # Unknown mask rank; let downstream throw a clearer error
    return attn_mask


def _build_mask(
    model_config: Dict[str, Any],
    # Newly added codes
    device: torch.device,
    dtype: torch.dtype,
    block_idx: int,
) -> torch.Tensor:
    """Build unified KV representation and attention mask for parallel processing using vectorized operations."""
    
    text_seq_len = model_config.get("text_seq_len", 512)
    image_size = model_config.get("image_size", (8192, 4096))
    patch_size = model_config.get("patch_size", 16)
    window_size = model_config.get("window_size", (1024, 512))
    
    H_p = image_size[0] // patch_size
    W_p = image_size[1] // patch_size
    window_patches_H = window_size[0] // patch_size
    window_patches_W = window_size[1] // patch_size
    joint_denoise_H = model_config.get("joint_denoise_size", (256, 256))[0] // patch_size
    joint_denoise_W = model_config.get("joint_denoise_size", (256, 256))[1] // patch_size
    
    # Shifted window logic: alternate between regular and shifted windows
    # Even layers (0, 2, 4, ...) use regular windows
    # Odd layers (1, 3, 5, ...) use shifted windows
    # shift_size_H = random.randint(0, window_patches_H // 2) if block_idx % 2 == 1 else 0
    # shift_size_W = random.randint(0, window_patches_W // 2) if block_idx % 2 == 1 else 0
    shift_size_H = 1
    shift_size_W = 1
    # Calculate number of windows
    h_windows = math.ceil(H_p / window_patches_H)
    w_windows = math.ceil(W_p / window_patches_W)
    
    hi_res_len = H_p * W_p
    low_res_len = joint_denoise_H * joint_denoise_W 
    total_q_len = text_seq_len + hi_res_len + low_res_len
    
    # Initialize attention mask
    attention_mask = torch.zeros(total_q_len, total_q_len, dtype=torch.bool, device=device)
    
    # =================== Text Attention Patterns ===================
    # Text tokens can attend based on configuration
    if model_config.get("text_attend_scale", "all") == "all":
        attention_mask[:text_seq_len, :] = True
    else:
        # attention_mask[:text_seq_len, :text_seq_len] = True  # text-to-text
        attention_mask[:text_seq_len, :-low_res_len] = True  # text-to-hi-res
    
    # All tokens can attend to text tokens
    attention_mask[:, :text_seq_len] = True
    
    # =================== Hi-Res Window Attention ===================
    # Check if sliding window attention is enabled
    use_sliding_window = model_config.get("use_sliding_window", False)
    
    # Create 2D coordinate mapping for hi-res tokens using vectorized operations
    h_indices = torch.arange(hi_res_len, device=device) // W_p
    w_indices = torch.arange(hi_res_len, device=device) % W_p
    
    # Define hi-res token boundaries (needed for both sliding and shifted window)
    hi_res_start = text_seq_len
    hi_res_end = text_seq_len + hi_res_len
    
    if use_sliding_window:
        # =================== Sliding Window Attention ===================
        # For sliding window, create a mask based on spatial distance
        # Each token can attend to tokens within a sliding window around it
        
        # Create coordinate matrices for all tokens
        h_coords = h_indices.unsqueeze(1)  # [hi_res_len, 1]
        w_coords = w_indices.unsqueeze(1)  # [hi_res_len, 1]
        
        # Create coordinate matrices for all other tokens
        h_coords_other = h_indices.unsqueeze(0)  # [1, hi_res_len]
        w_coords_other = w_indices.unsqueeze(0)  # [1, hi_res_len]
        
        # Calculate spatial distances
        h_dist = torch.abs(h_coords - h_coords_other)
        w_dist = torch.abs(w_coords - w_coords_other)
        
        # Define sliding window radius (half of window size)
        window_radius_H = window_patches_H // 2
        window_radius_W = window_patches_W // 2
        
        # Create sliding window mask
        sliding_window_mask = (h_dist <= window_radius_H) & (w_dist <= window_radius_W)

        # If tokens are permuted to window-first order, permute the mask accordingly
        if model_config.get("permute_window_first", False):
            # Compute window-first permutation (group by window, then raster-scan within each window)
            window_h_ids = h_indices // window_patches_H
            window_w_ids = w_indices // window_patches_W
            window_ids = window_h_ids * w_windows + window_w_ids
            within_h = h_indices % window_patches_H
            within_w = w_indices % window_patches_W
            within_idx = within_h * window_patches_W + within_w
            order_key = window_ids * (window_patches_H * window_patches_W) + within_idx
            perm = torch.argsort(order_key)

            # Reindex rows and columns to match permuted token order (P @ M @ P^T)
            sliding_window_mask = sliding_window_mask.index_select(0, perm).index_select(1, perm)

        # Apply the (possibly permuted) sliding window mask
        attention_mask[hi_res_start:hi_res_end, hi_res_start:hi_res_end] = sliding_window_mask
        
    else:
        # =================== Shifted Window Attention ===================
        # Apply shifting for odd layers
        if shift_size_H > 0 or shift_size_W > 0:
            h_indices = (h_indices + shift_size_H) % H_p
            w_indices = (w_indices + shift_size_W) % W_p
        
        # Assign window IDs using vectorized operations
        window_h_ids = h_indices // window_patches_H
        window_w_ids = w_indices // window_patches_W
        window_ids = window_h_ids * w_windows + window_w_ids
        
        # Use broadcasting to create window mask efficiently
        window_ids_query = window_ids.unsqueeze(1)  # [hi_res_len, 1]
        window_ids_key = window_ids.unsqueeze(0)    # [1, hi_res_len]
        same_window_mask = (window_ids_query == window_ids_key)  # [hi_res_len, hi_res_len]

        # For shifted windows, also allow attention to 4-neighbor windows (up, down, left, right)
        if shift_size_H > 0 or shift_size_W > 0:
            dh = torch.abs(window_h_ids.unsqueeze(1) - window_h_ids.unsqueeze(0))
            dw = torch.abs(window_w_ids.unsqueeze(1) - window_w_ids.unsqueeze(0))
            window_mask = (dh + dw) <= 1  # same window or 4-neighbors
        else:
            window_mask = same_window_mask

        # Assign the window mask to the appropriate section of the attention mask
        attention_mask[hi_res_start:hi_res_end, hi_res_start:hi_res_end] = window_mask

        
    
    # =================== Hi-Res to Low-Res Attention ===================
    if model_config.get("hi_res_attend_scale", "all") == "all":
        if low_res_len <= 256:
            attention_mask[hi_res_start:hi_res_end, -low_res_len:] = True
        else:
            window_area = window_patches_H * window_patches_W 
            # If tokens are permuted window-first, align HR windows to LR windows by window groups
            if model_config.get("permute_window_first", False):
                hi_windows_count = math.ceil(H_p / window_patches_H) * math.ceil(W_p / window_patches_W)
                lr_windows_count = math.ceil(low_res_len / window_area)
                if lr_windows_count <= 0:
                    lr_windows_count = 1
                group_size_windows = math.ceil(hi_windows_count / lr_windows_count)

                for i in range(lr_windows_count):
                    hi_win_start = i * group_size_windows
                    hi_win_end = min((i + 1) * group_size_windows, hi_windows_count)
                    if hi_win_start >= hi_win_end:
                        continue
                    hi_res_chunk_start = hi_res_start + hi_win_start * window_area
                    hi_res_chunk_end = hi_res_start + hi_win_end * window_area

                    low_res_window_start = hi_res_end + i * window_area
                    low_res_window_end = min(hi_res_end + (i + 1) * window_area, hi_res_end + low_res_len)

                    attention_mask[hi_res_chunk_start:hi_res_chunk_end,
                                   low_res_window_start:low_res_window_end] = True
            else:
                # Raster-order fallback: partition hi-res into equal sized chunks matching LR windows
                window_size = window_area
                num_low_res_windows = max(low_res_len // window_size, 1)
                hi_res_chunk_size = hi_res_len // num_low_res_windows

                for i in range(num_low_res_windows):
                    hi_res_chunk_start = hi_res_start + i * hi_res_chunk_size
                    hi_res_chunk_end = hi_res_start + (i + 1) * hi_res_chunk_size
                    low_res_window_start = hi_res_end + i * window_size
                    low_res_window_end = hi_res_end + (i + 1) * window_size
                    attention_mask[hi_res_chunk_start:hi_res_chunk_end,
                                   low_res_window_start:low_res_window_end] = True
                
    
    # =================== Low-Res Attention ===================
    # Low-res guidance tokens use block diagonal self-attention with 256x256 blocks
    if low_res_len > 256:
        block_size = 256
        num_blocks = (low_res_len + block_size - 1) // block_size
        
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, low_res_len)
            
            # Set diagonal block attention
            attention_mask[-low_res_len + block_start:-low_res_len + block_end, 
                          -low_res_len + block_start:-low_res_len + block_end] = True
    else:
        attention_mask[-low_res_len:, -low_res_len:] = True
    # attention_mask[-low_res_len:, -low_res_len:] = True
    # Convert boolean mask to float mask for attention
    # final_mask = torch.where(attention_mask, 0.0, -torch.inf).to(dtype)
    final_mask = attention_mask
    
    return final_mask


def _build_blocksparse_mask_thl(
    model_config: Dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    heads: int,
    text_seq_len: int,
    hi_res_len: int,
    low_res_len: int,
):
    """Build a THL block-sparse mask at 128-token granularity.

    Mask semantics (queries -> keys):
    - T queries: attend to T and H only.
    - H queries: attend to all T; H-H uses overlapping diagonal blocks (configurable block size and overlap);
      attend to all L or partitioned L based on config.
    - L queries: attend to L only.

    Returns:
    - base_blockmask: (num_q_blocks, blocksparse_head_num, num_k_blocks) uint8 0/1
    - head_mask_type: (nheads,) int32 with 1 meaning blocksparse pattern for each head
    - q_blocks, k_blocks: number of 128-token blocks along query/key axes
    """
    kernel_block = 128

    total_len = text_seq_len + hi_res_len + low_res_len
    q_blocks = (total_len + kernel_block - 1) // kernel_block
    k_blocks = q_blocks

    # Token ranges
    t_start, t_len = 0, text_seq_len
    h_start, h_len = text_seq_len, hi_res_len
    l_start, l_len = text_seq_len + hi_res_len, low_res_len

    def to_block_range(start_tok: int, length_tok: int) -> Tuple[int, int]: # eg. 512-4608 > 4-36
        if length_tok <= 0:
            return 0, -1
        start_b = start_tok // kernel_block
        end_b = (start_tok + length_tok - 1) // kernel_block
        return start_b, end_b

    t_b_s, t_b_e = to_block_range(t_start, t_len)
    h_b_s, h_b_e = to_block_range(h_start, h_len)
    l_b_s, l_b_e = to_block_range(l_start, l_len)

    base = torch.zeros((batch_size, heads, q_blocks, k_blocks), dtype=torch.uint8, device=device)

    patch_size = int(model_config.get("patch_size", 16))
    window_size = model_config.get("window_size", (256, 256))
    # window_size_blocks is not used in diagonal-overlap mode

    def set_rect(r0: int, r1: int, c0: int, c1: int):
        # turning on attention for rectangular regions in the block mask.
        if r0 > r1 or c0 > c1:
            return
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(q_blocks - 1, r1)
        c1 = min(k_blocks - 1, c1)
        if r0 <= r1 and c0 <= c1:
            base[:, :, r0 : r1 + 1, c0 : c1 + 1] = 1

    # T queries -> T and H
    if t_b_e >= t_b_s:
        if model_config.get("text_attend_scale", "limited") == "all":
            set_rect(t_b_s, t_b_e, 0, k_blocks-1)  # Text can see everything
        else:
            set_rect(t_b_s, t_b_e, 0, max(h_b_e, t_b_e))  # Text can see Text and Hi-res

    # H queries
    if h_b_e >= h_b_s:
        # H -> T (all hi-res tokens attend to all text)
        set_rect(h_b_s, h_b_e, 0, t_b_e)
        
        # H -> H (overlapping diagonal blocks only)
        overlap_ratio = float(model_config.get("bs_diag_overlap", 0.5))
        block_size_tokens = int(model_config.get("bs_diag_block_tokens", model_config.get("window_size", (256, 256))[0]))
        block_span_blocks = max((block_size_tokens + kernel_block - 1) // kernel_block, 1)
        step_blocks = max(int(block_span_blocks * (1.0 - overlap_ratio)), 1)
        hi_blocks = (h_b_e - h_b_s + 1)

        start_b = 0
        while start_b < hi_blocks:
            r0 = h_b_s + start_b
            r1 = min(h_b_s + start_b + block_span_blocks - 1, h_b_e)
            set_rect(r0, r1, r0, r1)
            start_b += step_blocks
        
        # H -> L (Handle differently based on hi_res_attend_scale and window size)
        if model_config.get("hi_res_attend_scale", "all") == "all":
            # Check if we need chunked attention based on window size comparison
            block_size = model_config.get("window_size", (256, 256))[0]
            if low_res_len <= block_size:
                # Simple case: all hi-res tokens attend to all low-res tokens
                set_rect(h_b_s, h_b_e, l_b_s, l_b_e)
            else:
                # Partitioned case: divide hi-res into chunks that attend to corresponding low-res windows
                num_low_res_windows = max(low_res_len // block_size, 1)
                hi_res_chunk_blocks = max((h_b_e - h_b_s + 1) // num_low_res_windows, 1)
                low_res_window_blocks = max((l_b_e - l_b_s + 1) // num_low_res_windows, 1)
                
                for i in range(num_low_res_windows):
                    hi_res_chunk_start = h_b_s + i * hi_res_chunk_blocks
                    hi_res_chunk_end = h_b_s + (i + 1) * hi_res_chunk_blocks
                    
                    low_res_window_start = l_b_s + i * low_res_window_blocks
                    low_res_window_end = l_b_s + (i + 1) * low_res_window_blocks
                    
                    set_rect(hi_res_chunk_start, hi_res_chunk_end, low_res_window_start, low_res_window_end)
        # If hi_res_attend_scale is not "all", no connections are added for H -> L

    # L queries -> L (and optionally -> T to match dense behavior)
    if l_b_e >= l_b_s:
        set_rect(l_b_s, l_b_e, l_b_s, l_b_e)  # Low-res tokens attend to themselves
        set_rect(l_b_s, l_b_e, t_b_s, t_b_e)  # Low-res tokens attend to text

    head_mask_type = torch.ones((heads,), dtype=torch.int32, device=device)
    return base, head_mask_type, q_blocks, k_blocks


logger = logging.getLogger(__name__) 
def _get_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None,):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)



def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    # Newly added codes
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    timestep: Optional[int] = None,
) -> torch.FloatTensor:
    """Modified attention forward with parallel window processing"""
    
    # If low_res_guidance is None, disable all LoRA
    if low_res_guidance is None:
        with enable_lora((attn,), False):
            return _attn_forward_impl(attn, hidden_states, image_rotary_emb, low_res_guidance, model_config, block_idx, encoder_hidden_states, timestep)
    else:
        return _attn_forward_impl(attn, hidden_states, image_rotary_emb, low_res_guidance, model_config, block_idx, encoder_hidden_states, timestep)

def _attn_forward_impl(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    timestep: Optional[int] = None,
) -> torch.FloatTensor:
    """Implementation of attention forward with parallel window processing"""
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    
    text_seq_len = model_config.get("text_seq_len", 512)
    
    # =============================== Project Hi-Res tokens ===============================
    # Project Q, K, V with LoRA
    # NOTE: when `attn.fused_projections` is enabled, q/k/v are produced by `attn.to_qkv`.
    # We must toggle LoRA on the *actual* projection module used, otherwise "LoRA disabled"
    # can silently leak into the no-LR (base) path.
    if getattr(attn, "fused_projections", False) and hasattr(attn, "to_qkv"):
        qkv_modules = (attn.to_qkv,)
    else:
        qkv_modules = (attn.to_q, attn.to_k, attn.to_v)

    with enable_lora(qkv_modules, False):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if attn.added_kv_proj_dim is not None:
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

    # =============================== Project Low-Res tokens ===============================
    # Project Q, K, V w/o LoRA
    if low_res_guidance is not None:
        with enable_lora(qkv_modules, True):
            low_res_guidance_query, low_res_guidance_key, low_res_guidance_value, _, _, _ = _get_qkv_projections(
                attn, low_res_guidance, None)

        low_res_guidance_query = low_res_guidance_query.unflatten(-1, (attn.heads, -1))
        low_res_guidance_key = low_res_guidance_key.unflatten(-1, (attn.heads, -1))
        low_res_guidance_value = low_res_guidance_value.unflatten(-1, (attn.heads, -1))

        low_res_guidance_query = attn.norm_q(low_res_guidance_query)
        low_res_guidance_key = attn.norm_k(low_res_guidance_key)
        query = torch.cat([query, low_res_guidance_query], dim=1)
        key = torch.cat([key, low_res_guidance_key], dim=1)
        value = torch.cat([value, low_res_guidance_value], dim=1)

    # =============================== Build Unified KV and Mask ===============================
    # If we're using block-sparse kernels, DO NOT build the dense boolean mask to save memory.
    use_blocksparse = (
        BLOCK_SPARSE_ATTENTION_AVAILABLE and model_config.get("use_block_sparse_attention", False)
    )
    use_sage = (
        SAGE_BLOCKSPARSE_AVAILABLE and model_config.get("use_sage_blocksparse", False)
    )
    if low_res_guidance is not None and (not use_blocksparse) and (not use_sage):
        # Optionally load a precomputed dense mask for speed (must match current token lengths).
        # NOTE: previously this was hardcoded to a user-specific path, which breaks portability.
        precomputed_path = model_config.get("precomputed_attn_mask_path", "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/mask_expanded_128x64_test-new.pt")
        global _PRECOMPUTED_DENSE_MASK
        if precomputed_path is not None and os.path.exists(precomputed_path):
            if _PRECOMPUTED_DENSE_MASK is None:
                _PRECOMPUTED_DENSE_MASK = torch.load(precomputed_path, map_location="cpu")
            attn_mask = _PRECOMPUTED_DENSE_MASK
        else:
            attn_mask = _build_mask(
                model_config=model_config,
                device=query.device,
                dtype=query.dtype,
                block_idx=block_idx,
            )
    else:
        attn_mask = None    

    # Apply position embeddings to queries
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # =============================== Perform Parallel Attention ===============================
    if low_res_guidance is not None:
        # SDPA / blocksparse paths expect [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if use_sage:
            nheads = query.shape[1]
            cpu_mask_key = (
                "SAGE_THL_128x64_mask",
                int(batch_size),
                int(nheads),
                int(model_config.get("image_size", (4096, 4096))[0]),
                int(model_config.get("image_size", (4096, 4096))[1]),
            )
            gpu_mask_key = (cpu_mask_key, str(query.device))
            cached_gpu = _SAGE_BLOCK_MASK_CACHE.get(gpu_mask_key)  # type: ignore[arg-type]
            if cached_gpu is not None:
                block_map = cached_gpu
            else:
                block_map_cpu = _SAGE_BLOCK_MASK_CACHE.get(cpu_mask_key)  # type: ignore[arg-type]
                if block_map_cpu is None:
                    block_map_cpu = _load_sage_mask_thl_128x64(
                        model_config,
                        batch_size=int(batch_size),
                        heads=int(nheads),
                    )
                    _SAGE_BLOCK_MASK_CACHE[cpu_mask_key] = block_map_cpu
                block_map = (
                    block_map_cpu.to(device=query.device, dtype=torch.uint8, non_blocking=True)
                    .contiguous()
                )
                _SAGE_BLOCK_MASK_CACHE[gpu_mask_key] = block_map

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            hidden_states = _call_sage_blocksparse(query, key, value, block_map, model_config)
        elif FLASH_ATTENTION_AVAILABLE and model_config.get("use_flash_attention", False):
            if model_config.get("proportional_attn", False):
                l1 = model_config.get("window_size", [256, 256])[0]*model_config.get("window_size", [256, 256])[1]//16//16 + model_config.get("text_seq_len", 512) +model_config.get("joint_denoise_size", [640, 640])[0]*model_config.get("joint_denoise_size", [640, 640])[1]//16//16 
                l2 = 4096 + 512
                head_dim = query.shape[-1]
                scale = math.sqrt(math.log(l1, l2) / head_dim)
                attn.scale = scale
                # print(f"Proportional attention scale: {scale}")
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=_normalize_attn_mask_for_scores(attn_mask, query.shape[2], key.shape[2], query.device),
                dropout_p=0.0,
                scale=attn.scale if model_config.get("proportional_attn", False) else None,
            )
        # Option 2: Standard attention (fallback)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
            if attn_mask is not None:
                attn_mask = _normalize_attn_mask_for_scores(attn_mask, scores.shape[-2], scores.shape[-1], scores.device)
                if attn_mask.dtype == torch.bool:
                    # score_copy = scores.clone()
                    scores = scores.masked_fill(attn_mask.logical_not(), -torch.inf)
                else:
                    scores = scores + attn_mask
            attn_probs = F.softmax(scores, dim=-1)
            hidden_states = torch.matmul(attn_probs, value)
        # back to [batch, seq, heads, head_dim]
        hidden_states = hidden_states.transpose(1, 2)
    else:
        # dispatch_attention_fn path expects [batch, seq, heads, head_dim]
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=_normalize_attn_mask_for_scores(attn_mask, query.shape[1], key.shape[1], query.device) if attn_mask is not None else None,
            # backend=self._attention_backend,
            # parallel_config=self._parallel_config,
        )

    # Reshape output
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    # Apply output projection
    if encoder_hidden_states is not None:
        # Split text and image outputs
        encoder_output = hidden_states[:, :text_seq_len, :]
        if low_res_guidance is not None:
            low_res_hidden_states = hidden_states[:, -low_res_guidance_key.shape[1]:, :]
            hidden_states_output = hidden_states[:, text_seq_len:-low_res_guidance_key.shape[1], :]
        else:
            hidden_states_output = hidden_states[:, text_seq_len:, :]

        with enable_lora((attn.to_out[0], attn.to_out[1]), False):
            hidden_states_output = attn.to_out[0](hidden_states_output)
            hidden_states_output = attn.to_out[1](hidden_states_output)

        encoder_output = attn.to_add_out(encoder_output)

        if low_res_guidance is not None:
            with enable_lora((attn.to_out[0], attn.to_out[1]), True):
                low_res_hidden_states = attn.to_out[0](low_res_hidden_states)
                low_res_hidden_states = attn.to_out[1](low_res_hidden_states)
            return hidden_states_output, encoder_output, low_res_hidden_states
        return hidden_states_output, encoder_output, None
    else:
        if low_res_guidance is not None:
            hidden_states_output = hidden_states[:, :-low_res_guidance_key.shape[1], :]
            low_res_hidden_states = hidden_states[:, -low_res_guidance_key.shape[1]:, :]
            return hidden_states_output, low_res_hidden_states
        else:
            return hidden_states, None

def single_attn_forward(
    attn,
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int = 0,
    timestep: Optional[int] = None,
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    """Attention forward for single transformer blocks using Flux2ParallelSelfAttention.

    Unlike attn_forward (designed for Flux2Attention with separate to_q/to_k/to_v),
    this handles Flux2ParallelSelfAttention which fuses QKV + MLP into a single
    projection (to_qkv_mlp_proj).

    Returns (output, low_res_output) where low_res_output is None when low_res_guidance is None.
    """
    if low_res_guidance is None:
        # No LR branch should use the original (base) parameters.
        # Flux2ParallelSelfAttention uses fused projections (to_qkv_mlp_proj) and output projection (to_out),
        # both of which can be LoRA-injected by our config. Explicitly disable LoRA here.
        with enable_lora((attn.to_qkv_mlp_proj, attn.to_out), False):
            output = attn(hidden_states=hidden_states, image_rotary_emb=image_rotary_emb)
        return output, None

    lr_seq_len = low_res_guidance.shape[1]

    with enable_lora((attn.to_qkv_mlp_proj,), False):
        projected = attn.to_qkv_mlp_proj(hidden_states)
    qkv, mlp_hidden_states = torch.split(
        projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
    )
    query, key, value = qkv.chunk(3, dim=-1)

    with enable_lora((attn.to_qkv_mlp_proj,), True):
        lr_projected = attn.to_qkv_mlp_proj(low_res_guidance)
    lr_qkv, lr_mlp_hidden_states = torch.split(
        lr_projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
    )
    lr_query, lr_key, lr_value = lr_qkv.chunk(3, dim=-1)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    lr_query = lr_query.unflatten(-1, (attn.heads, -1))
    lr_key = lr_key.unflatten(-1, (attn.heads, -1))
    lr_value = lr_value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query)
    key = attn.norm_k(key)
    lr_query = attn.norm_q(lr_query)
    lr_key = attn.norm_k(lr_key)

    query = torch.cat([query, lr_query], dim=1)
    key = torch.cat([key, lr_key], dim=1)
    value = torch.cat([value, lr_value], dim=1)

    use_sage = (
        SAGE_BLOCKSPARSE_AVAILABLE and model_config.get("use_sage_blocksparse", False)
    )
    if not use_sage:
        precomputed_path = model_config.get("precomputed_attn_mask_path", None)
        global _PRECOMPUTED_DENSE_MASK
        if precomputed_path is not None and os.path.exists(precomputed_path):
            if _PRECOMPUTED_DENSE_MASK is None:
                _PRECOMPUTED_DENSE_MASK = torch.load(precomputed_path, map_location="cpu")
            attn_mask = _PRECOMPUTED_DENSE_MASK
        else:
            attn_mask = _build_mask(
                model_config=model_config,
                device=query.device,
                dtype=query.dtype,
                block_idx=block_idx,
            )
    else:
        attn_mask = None

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if use_sage:
        batch_size = hidden_states.shape[0]
        nheads = query.shape[1]
        cpu_mask_key = (
            "SAGE_THL_128x64_mask",
            int(batch_size),
            int(nheads),
            int(model_config.get("image_size", (4096, 4096))[0]),
            int(model_config.get("image_size", (4096, 4096))[1]),
        )
        gpu_mask_key = (cpu_mask_key, str(query.device))
        cached_gpu = _SAGE_BLOCK_MASK_CACHE.get(gpu_mask_key)  # type: ignore[arg-type]
        if cached_gpu is not None:
            block_map = cached_gpu
        else:
            block_map_cpu = _SAGE_BLOCK_MASK_CACHE.get(cpu_mask_key)  # type: ignore[arg-type]
            if block_map_cpu is None:
                block_map_cpu = _load_sage_mask_thl_128x64(
                    model_config,
                    batch_size=int(batch_size),
                    heads=int(nheads),
                )
                _SAGE_BLOCK_MASK_CACHE[cpu_mask_key] = block_map_cpu
            block_map = (
                block_map_cpu.to(device=query.device, dtype=torch.uint8, non_blocking=True)
                .contiguous()
            )
            _SAGE_BLOCK_MASK_CACHE[gpu_mask_key] = block_map
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_output = _call_sage_blocksparse(query, key, value, block_map, model_config)
    elif FLASH_ATTENTION_AVAILABLE and model_config.get("use_flash_attention", True):
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=_normalize_attn_mask_for_scores(attn_mask, query.shape[2], key.shape[2], query.device),
            dropout_p=0.0,
        )
    else:
        head_dim = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn_mask = _normalize_attn_mask_for_scores(attn_mask, scores.shape[-2], scores.shape[-1], scores.device)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.logical_not(), -torch.inf)
            else:
                scores = scores + attn_mask
        attn_probs = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, value)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.flatten(2, 3)
    attn_output = attn_output.to(query.dtype)

    main_attn_output = attn_output[:, :-lr_seq_len, :]
    lr_attn_output = attn_output[:, -lr_seq_len:, :]

    mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
    lr_mlp_hidden_states = attn.mlp_act_fn(lr_mlp_hidden_states)

    with enable_lora((attn.to_out,), False):
        main_output = attn.to_out(torch.cat([main_attn_output, mlp_hidden_states], dim=-1))
    with enable_lora((attn.to_out,), True):
        lr_output = attn.to_out(torch.cat([lr_attn_output, lr_mlp_hidden_states], dim=-1))

    return main_output, lr_output


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        return mod

    @staticmethod
    # split inside the transformer blocks, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
    def split(mod: torch.Tensor, mod_param_sets: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))


# Keep block_forward and single_block_forward unchanged as they just call attn_forward
def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    # Newly added codes
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    timestep: Optional[int] = None,
    double_stream_mod_img: Optional[torch.FloatTensor] = None,
    double_stream_mod_txt: Optional[torch.FloatTensor] = None,
    double_stream_mod_lr: Optional[torch.FloatTensor] = None,
):
    # If low_res_temb is not present, disable all LoRA
    if low_res_guidance is None:
        with enable_lora((self,), False):
            return _block_forward_impl(self, hidden_states, encoder_hidden_states, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep, double_stream_mod_img, double_stream_mod_txt, double_stream_mod_lr)
    else:
        return _block_forward_impl(self, hidden_states, encoder_hidden_states, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep, double_stream_mod_img, double_stream_mod_txt, double_stream_mod_lr)


def _block_forward_impl(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    timestep: Optional[int] = None,
    double_stream_mod_img: Optional[torch.FloatTensor] = None,
    double_stream_mod_txt: Optional[torch.FloatTensor] = None,
    double_stream_mod_lr: Optional[torch.FloatTensor] = None,
):
    # hi-res tokens need not to be finetuned
    # modulation param shape: [1, 1, self.dim]
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(double_stream_mod_img, 2)
    (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            double_stream_mod_txt, 2
        )
    if low_res_guidance is not None:
        (lr_shift_msa, lr_scale_msa, lr_gate_msa), (lr_shift_mlp, lr_scale_mlp, lr_gate_mlp) = Flux2Modulation.split(double_stream_mod_lr, 2)
    
    # Img stream
    with enable_lora((self.norm1,), False):
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

    # Text stream
    norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
    norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

    # Low-res stream
    if low_res_guidance is not None:
        with enable_lora((self.norm1,), True):
            norm_low_res_guidance = self.norm1(low_res_guidance)
            norm_low_res_guidance = (1 + lr_scale_msa) * norm_low_res_guidance + lr_shift_msa

    # Attention
    attn_forward_output = attn_forward(
        self.attn,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        low_res_guidance=None if low_res_guidance is None else norm_low_res_guidance,
        model_config=model_config,
        block_idx=block_idx,
        encoder_hidden_states=norm_encoder_hidden_states,
        timestep=timestep,
    )
    
    if low_res_guidance is None:
        attn_output, context_attn_output, _ = attn_forward_output
    else:   
        attn_output, context_attn_output, low_res_attn_output = attn_forward_output

    # Process attention outputs
    attn_output = gate_msa * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (1 + scale_mlp) * norm_hidden_states + shift_mlp

    if low_res_guidance is not None:
        low_res_attn_output = lr_gate_msa * low_res_attn_output
        low_res_guidance = low_res_guidance + low_res_attn_output

        norm_low_res_guidance = self.norm2(low_res_guidance)
        norm_low_res_guidance = (1 + lr_scale_mlp) * norm_low_res_guidance + lr_shift_mlp

    if context_attn_output is not None:
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_mlp) * norm_encoder_hidden_states + c_shift_mlp

    # Feed-forward
    with enable_lora((self.ff.linear_in, self.ff.linear_out), False):
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
    if low_res_guidance is not None:
        with enable_lora((self.ff.linear_in, self.ff.linear_out), True):
            lr_ff_output = self.ff(norm_low_res_guidance)
            lr_ff_output = lr_gate_mlp * lr_ff_output

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp * context_ff_output

    # Process outputs
    hidden_states = hidden_states + ff_output
    if low_res_guidance is not None:
        low_res_guidance = low_res_guidance + lr_ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return (encoder_hidden_states, hidden_states, low_res_guidance) if low_res_guidance is not None else (encoder_hidden_states, hidden_states, None)

def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    low_res_temb: Optional[torch.FloatTensor],  
    image_rotary_emb: Optional[torch.FloatTensor],
    # Newly added codes
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    timestep: Optional[int] = None,
    double_stream_mod_img: Optional[torch.FloatTensor] = None,
    double_stream_mod_txt: Optional[torch.FloatTensor] = None,
    double_stream_mod_lr: Optional[torch.FloatTensor] = None,
):
    # If low_res_temb is not present, disable all LoRA
    if low_res_temb is None:
        with enable_lora((self,), False):
            return _single_block_forward_impl(self, hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep, double_stream_mod_img, double_stream_mod_txt, double_stream_mod_lr)
    else:
        return _single_block_forward_impl(self, hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep, double_stream_mod_img, double_stream_mod_txt, double_stream_mod_lr)


def _single_block_forward_impl(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    low_res_temb: Optional[torch.FloatTensor],
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = {},
    block_idx: int=0,
    timestep: Optional[int] = None,
    double_stream_mod_img: Optional[torch.FloatTensor] = None,
    double_stream_mod_txt: Optional[torch.FloatTensor] = None,
    double_stream_mod_lr: Optional[torch.FloatTensor] = None,
):
    # If low_res_temb is not present, disable all LoRA
    mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb, 1)[0]
    norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

    if low_res_guidance is not None:
        lr_mod_shift, lr_mod_scale, lr_mod_gate = Flux2Modulation.split(low_res_temb, 1)[0]

        norm_low_res_guidance = self.norm(low_res_guidance)
        norm_low_res_guidance = (1 + lr_mod_scale) * norm_low_res_guidance + lr_mod_shift

    attn_output, lr_attn_output = single_attn_forward(
        self.attn,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        low_res_guidance=None if low_res_guidance is None else norm_low_res_guidance,
        model_config=model_config,
        block_idx=block_idx,
        timestep=timestep,
    )

    hidden_states = hidden_states + mod_gate * attn_output

    if low_res_guidance is not None:
        low_res_guidance = low_res_guidance + lr_mod_gate * lr_attn_output

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if low_res_guidance is not None and low_res_guidance.dtype == torch.float16:
        low_res_guidance = low_res_guidance.clip(-65504, 65504)

    return hidden_states, low_res_guidance if low_res_guidance is not None else None

    