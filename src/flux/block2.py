import os
import torch
from typing import Optional, Dict, Any, Tuple
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
try: 
    from .lora_controller import enable_lora
except:
    from lora_controller import enable_lora

import math
from diffusers.models.embeddings import apply_rotary_emb
import random
import logging
import torch.nn as nn

# Sage / Sparge blocksparse attention (CUDA)
SAGE_CORE_AVAILABLE = False
try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
    SAGE_CORE_AVAILABLE = True
except Exception:
    print("### WARNING: Sage / Sparge blocksparse attention (CUDA) is not available")
    pass
SAGE_BLOCKSPARSE_AVAILABLE = SAGE_CORE_AVAILABLE

def _call_sage_blocksparse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: Optional[torch.Tensor],
    attn: Attention,
    model_config: Dict[str, Any],
):
    if SAGE_CORE_AVAILABLE:
        return block_sparse_sage2_attn_cuda(
            query, key, value,
            mask_id=block_map,
            dropout_p=0.0,
            scale=query.shape[-1]**-0.5,
            smooth_k=bool(model_config.get("sage_smooth_k", True)),
            pvthreshd=int(model_config.get("sage_pvthreshd", 50)),
            attention_sink=bool(model_config.get("sage_attention_sink", False)),
            tensor_layout="HND",
            output_dtype=query.dtype,
            return_sparsity=False,
        )
    raise RuntimeError("Sage blocksparse attention is not available")


# Cache for block-sparse masks to avoid rebuilding every step
_BLOCK_MASK_CACHE: Dict[Tuple, torch.BoolTensor] = {}

# Print-once cache for FLOP/GFLOP summaries (avoid spamming logs)
_GFLOP_PRINTED: set = set()


def _maybe_print_sage_gflops(
    *,
    block_map: torch.Tensor,
    head_dim: int,
    block_q: int,
    block_k: int,
    proj_in_dim: Optional[int] = None,
    model_config: Dict[str, Any],
    tag: str = "SAGE-THL",
) -> None:
    """
    Print estimated attention matmul FLOPs/GFLOPs for Sage block-sparse vs dense (Flash/SDPA).

    We count only the two matmuls in attention:
      QK^T and P@V => total FLOPs ~= 4 * head_dim * (#token_pairs over B,H,q,k)

    For block masks, token_pairs ~= nnz_blocks * block_q * block_k.

    Controlled by:
      model_config['print_gflops'] = True
    """
    if not model_config.get("print_gflops", False):
        return
    if block_map is None:
        return
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)

    # block_map is expected (B, H, Q_blocks, K_blocks)
    if block_map.dim() != 4:
        return

    B, H, Qb, Kb = map(int, block_map.shape)
    inner_dim = int(H * head_dim)
    # In (self-)attention with asymmetric block sizes, Lq and Lk should still match.
    Lq = int(Qb * block_q)
    Lk = int(Kb * block_k)
    L = Lq if Lq == Lk else None

    # Print once per distinct shape/config
    img_h, img_w = model_config.get("image_size", (None, None))
    key = (tag, img_h, img_w, B, H, Qb, Kb, int(head_dim), int(block_q), int(block_k), int(proj_in_dim or -1))
    if key in _GFLOP_PRINTED:
        return
    _GFLOP_PRINTED.add(key)

    nnz_blocks = int(block_map.sum().item())
    total_blocks = int(block_map.numel())
    density_blocks = (nnz_blocks / total_blocks) if total_blocks else 0.0

    active_pairs = nnz_blocks * block_q * block_k
    total_pairs = total_blocks * block_q * block_k
    density_pairs = (active_pairs / total_pairs) if total_pairs else 0.0

    # Matmul-only attention FLOPs:
    #   dense: 4 * head_dim * (B*H*Lq*Lk)  where (B*H*Lq*Lk) == total_pairs
    #   sparse: 4 * head_dim * active_pairs
    sparse_flops = int(4 * head_dim * active_pairs)
    dense_flops = int(4 * head_dim * total_pairs)

    sparse_gflops = sparse_flops / 1e9
    dense_gflops = dense_flops / 1e9
    speedup = (dense_flops / sparse_flops) if sparse_flops else float("inf")

    # "Full attention" (approx): QKV projections + (QK + PV) + output projection.
    # - QKV: 3 * (2 * B * L * proj_in_dim * inner_dim)
    # - Out: 1 * (2 * B * L * inner_dim * proj_in_dim)
    # Projections are the same for sparse vs dense (same L); only the matmul term changes.
    full_sparse_gflops = None
    full_dense_gflops = None
    full_speedup = None
    if proj_in_dim is not None and L is not None and proj_in_dim > 0:
        proj_flops = int(3 * 2 * B * L * proj_in_dim * inner_dim)
        out_flops = int(2 * B * L * inner_dim * proj_in_dim)
        full_sparse_flops = proj_flops + sparse_flops + out_flops
        full_dense_flops = proj_flops + dense_flops + out_flops
        full_sparse_gflops = full_sparse_flops / 1e9
        full_dense_gflops = full_dense_flops / 1e9
        full_speedup = (full_dense_flops / full_sparse_flops) if full_sparse_flops else float("inf")

    print(
        f"[GFLOPs] {tag} img_size={img_h}x{img_w} "
        f"B={B} H={H} head_dim={head_dim} inner_dim={inner_dim} "
        f"block={block_q}x{block_k} "
        f"blocks nnz={nnz_blocks:,}/{total_blocks:,} ({100*density_blocks:.2f}%) "
        f"token-pairs density={100*density_pairs:.2f}% "
        f"matmul-only: sparse={sparse_gflops:.3f} GFLOPs, dense(Flash/SDPA)={dense_gflops:.3f} GFLOPs, speedup={speedup:.2f}x"
    )
    if full_sparse_gflops is not None and full_dense_gflops is not None and full_speedup is not None:
        print(
            f"[GFLOPs] {tag} full-attn(approx proj+matmul+out): "
            f"proj_in_dim={proj_in_dim} L={L} "
            f"sparse={full_sparse_gflops:.3f} GFLOPs, dense(Flash/SDPA)={full_dense_gflops:.3f} GFLOPs, speedup={full_speedup:.2f}x"
        )


def _load_blocksparse_mask_thl_128x64(
    model_config: Dict[str, Any],
    *,
    batch_size: int,
    heads: int,
) -> torch.BoolTensor:
    """
    THL mask on Sage's asymmetric grid:
      - Q-block = 128 tokens
      - K-block = 64 tokens

    Swin-style with 4-neighbor attention:
      - Each Hi-Res window attends to itself and 4 neighboring windows
      - Hi-Res to Low-Res: each HR window attends to 256 tokens in LR

    Returns:
      (B, H, Qk, Kk) bool
    """
    
    # Sage defaults
    model_config.setdefault("use_sage_blocksparse", True)
    model_config.setdefault("sage_kernel_blk_q", 128)
    model_config.setdefault("sage_kernel_blk_k", 64)
    model_config.setdefault("sage_smooth_k", True)
    model_config.setdefault("sage_pvthreshd", 50)
    model_config.setdefault("sage_attention_sink", False)

    # Resolve image size and build a minimal cache key
    img_h, img_w = model_config.get("image_size", (4096, 4096))
    cache_key = ("THL_128x64_mask", batch_size, heads, img_h, img_w)
    cached = _BLOCK_MASK_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    try:
        # Precomputed mask file paths keyed by image size (H, W)
        mask_path_6k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(6144, 6144)x(6144, 6144)_downsampled_128_test-window.pt"
        mask_path_4k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(4096, 4096)x(4096, 4096)_downsampled_128_test-window.pt"
        mask_path_2k_6k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(2048, 6144)x(2048, 6144)_downsampled_128_test-window.pt"
        mask_path_1k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(1024, 1024)x(1024, 1024)_downsampled_128_test-window.pt"
        mask_path_2k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(2048, 2048)x(2048, 2048)_downsampled_128_test-window.pt"
        mask_path_3k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(3072, 3072)x(3072, 3072)_downsampled_128_test-window.pt"
        mask_path_8k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(8192, 8192)x(8192, 8192)_downsampled_128_test-window.pt"
        mask_path_4k_8k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(4096, 8192)x(4096, 8192)_downsampled_128_test-window.pt"
        mask_path_6k_4k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(6144, 4096)x(6144, 4096)_downsampled_128_test-window.pt"
        mask_path_4k_3k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(4096, 3072)x(4096, 3072)_downsampled_128_test-window.pt"
        mask_path_8k_6k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(8192, 6144)x(8192, 6144)_downsampled_128_test-window.pt"
        mask_path_4k_2k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(4096, 2048)x(4096, 2048)_downsampled_128_test-window.pt"
        mask_path_8k_4k = "/scratch/yuyao/Scale-DiT/src/flux/attn_masks/(8192, 4096)x(8192, 4096)_downsampled_128_test-window.pt"
        size_to_path = {
            (4096, 4096): mask_path_4k,
            (2048, 6144): mask_path_2k_6k,
            (6144, 6144): mask_path_6k,
            (1024, 1024): mask_path_1k,
            (2048, 2048): mask_path_2k,
            (3072, 3072): mask_path_3k,
            (8192, 8192): mask_path_8k,
            (4096,8192): mask_path_4k_8k,
            (6144,4096): mask_path_6k_4k,
            (4096,3072): mask_path_4k_3k,
            (8192, 6144): mask_path_8k_6k,
            (4096, 2048): mask_path_4k_2k,
            (8192, 4096): mask_path_8k_4k,
        }
        mask_path = size_to_path.get((img_h, img_w))
        if mask_path is None:
            # Preserve original error message style (height-focused)
            raise ValueError(f"Image size {img_h} not supported, please build the mask first.")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Block mask file does not exist: {mask_path}")
        # Map to CPU to avoid device-id mismatches (e.g., saved with cuda:4, current rank has cuda:0)
        block_mask = torch.load(mask_path, map_location="cpu")
        if isinstance(block_mask, torch.Tensor):
            block_mask = block_mask.to(torch.bool)
        # Expand to (B, H, L, L) format if needed
        if block_mask.dim() == 2:
            # (L, L) -> (B, H, L, L)
            block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, heads, -1, -1)
        elif block_mask.dim() == 3:
            # (H, L, L) -> (B, H, L, L)
            block_mask = block_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif block_mask.dim() == 4:
            # Already (B, H, L, L) format
            if block_mask.shape[0] != batch_size or block_mask.shape[1] != heads:
                block_mask = block_mask.expand(batch_size, heads, -1, -1)
        print(f"Loaded block mask from: {mask_path}")
        # Cache the loaded/expanded mask to avoid re-loading from disk next time
        _BLOCK_MASK_CACHE[cache_key] = block_mask
        return block_mask
    except Exception as e:
        raise Exception(f"Failed to load block mask: {e}")

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
    # Toggle LoRA on the *actual* projection module used.
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
    
    # Apply position embeddings to queries
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # =============================== Perform Parallel Attention ===============================
    if low_res_guidance is not None:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if SAGE_BLOCKSPARSE_AVAILABLE and model_config.get("use_sage_blocksparse", False):
            nheads = query.shape[1]
            head_dim = query.shape[3]
            # Load CPU mask (cached by shape), then cache a GPU-resident copy by device.
            img_h, img_w = model_config.get("image_size", (4096, 4096))
            cpu_mask_key = ("THL_128x64_mask", int(batch_size), int(nheads), int(img_h), int(img_w))
            gpu_mask_key = (cpu_mask_key, str(query.device))
            cached_gpu = _BLOCK_MASK_CACHE.get(gpu_mask_key)
            if cached_gpu is not None:
                block_map = cached_gpu
            else:
                block_map_cpu = _BLOCK_MASK_CACHE.get(cpu_mask_key)
                if block_map_cpu is None:
                    block_map_cpu = _load_blocksparse_mask_thl_128x64(
                        model_config,
                        batch_size=batch_size,
                        heads=nheads,
                    )
                # Per SpargeAttn API: mask_id is 0/1 with shape (B,H,ceil(N/128),ceil(N/64)).
                # Keep a compact dtype to reduce bandwidth; ensure contiguous.
                block_map = block_map_cpu.to(device=query.device, dtype=torch.uint8, non_blocking=True).contiguous()
                _BLOCK_MASK_CACHE[gpu_mask_key] = block_map

            # Sage expects (B,H,N,D) or (N,H,D) with tensor_layout flag; we use HND here.
            # Ensure contiguous Q/K/V for the custom CUDA kernel fast path.
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            _maybe_print_sage_gflops(
                block_map=block_map,
                head_dim=head_dim,
                block_q=int(model_config.get("sage_kernel_blk_q", 128)),
                block_k=int(model_config.get("sage_kernel_blk_k", 64)),
                proj_in_dim=int(getattr(getattr(attn, "to_q", None), "in_features", 0)) or None,
                model_config=model_config,
                tag="SAGE-THL",
            )
            out = _call_sage_blocksparse(
                query, key, value,
                block_map,
                attn,
                model_config,
            )

            hidden_states = out
        else:
            raise RuntimeError("Sage blocksparse attention is required but not available. Please enable use_sage_blocksparse in model_config.")
        hidden_states = hidden_states.transpose(1, 2)
    else:
        # Transpose to (B, H, N, D) format expected by dispatch_attention_fn
        query = query#.transpose(1, 2)
        key = key#.transpose(1, 2)
        value = value#.transpose(1, 2)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
        )
        # Transpose back to (B, N, H, D) for consistency with reshape below
        hidden_states = hidden_states#.transpose(1, 2)

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

    batch_size = hidden_states.shape[0]
    lr_seq_len = low_res_guidance.shape[1]

    # Match v1 single-block LoRA semantics:
    # - If LR guidance is present, do not force-disable LoRA for the main fused projection.
    # - LR guidance projection stays explicitly LoRA-enabled.
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

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if SAGE_BLOCKSPARSE_AVAILABLE and model_config.get("use_sage_blocksparse", False):
        nheads = query.shape[1]


        img_h, img_w = model_config.get("image_size", (4096, 4096))
        cpu_mask_key = ("THL_128x64_mask", int(batch_size), int(nheads), int(img_h), int(img_w))
        gpu_mask_key = (cpu_mask_key, str(query.device))
        cached_gpu = _BLOCK_MASK_CACHE.get(gpu_mask_key)
        if cached_gpu is not None:
            block_map = cached_gpu
        else:
            block_map_cpu = _BLOCK_MASK_CACHE.get(cpu_mask_key)
            if block_map_cpu is None:
                block_map_cpu = _load_blocksparse_mask_thl_128x64(
                    model_config, batch_size=batch_size, heads=nheads,
                )
            block_map = block_map_cpu.to(device=query.device, dtype=torch.uint8, non_blocking=True).contiguous()
            _BLOCK_MASK_CACHE[gpu_mask_key] = block_map

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        attn_output = block_sparse_sage2_attn_cuda(
            query, key, value,
            mask_id=block_map,
            dropout_p=0.0,
            scale=None,
            smooth_k=bool(model_config.get("sage_smooth_k", True)),
            pvthreshd=int(model_config.get("sage_pvthreshd", 50)),
            attention_sink=bool(model_config.get("sage_attention_sink", False)),
            tensor_layout="HND",
            output_dtype=query.dtype,
            return_sparsity=False,
        )
    else:
        raise RuntimeError("Sage blocksparse attention is required but not available for single_attn_forward with low_res_guidance.")

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

    