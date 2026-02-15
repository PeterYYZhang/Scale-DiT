import os
import torch
from typing import Optional, Dict, Any, Tuple
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
try: 
    from .lora_controller import enable_lora
except:
    from lora_controller import enable_lora

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
            scale=attn.scale,
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

def _attn_forward_impl(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = None,
    block_idx: int=0,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    timestep: Optional[int] = None,
) -> torch.FloatTensor:
    """Implementation of attention forward with parallel window processing"""
    if model_config is None:
        model_config = {}

    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    

    text_seq_len = model_config.get("text_seq_len", 512)
    
    # =============================== Project Hi-Res tokens ===============================
    # Project Q, K, V with LoRA
    if low_res_guidance is not None:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    else:
        with enable_lora((attn.to_q, attn.to_k, attn.to_v), False):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # =============================== Project Context tokens ===============================
    # Handle encoder hidden states
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    # =============================== Project Low-Res tokens ===============================
    # Project Q, K, V w/o LoRA
    if low_res_guidance is not None:
        use_lr_qkv_cache = bool(model_config.get("cache_low_res_guidance_qkv", False)) and bool(
            model_config.get("low_res_guidance_as_condition", False)
        )
        if use_lr_qkv_cache:
            cache_key = (
                id(low_res_guidance),
                tuple(low_res_guidance.shape),
                str(low_res_guidance.dtype),
                str(low_res_guidance.device),
                int(attn.heads),
                int(head_dim),
            )
            cached = getattr(attn, "_lr_cond_qkv_cache", None)
            if cached is not None and cached.get("key") == cache_key:
                low_res_guidance_query, low_res_guidance_key, low_res_guidance_value = cached["value"]
            else:
                with enable_lora((attn.to_q, attn.to_k, attn.to_v), True):
                    low_res_guidance_query = attn.to_q(low_res_guidance)
                    low_res_guidance_key = attn.to_k(low_res_guidance)
                    low_res_guidance_value = attn.to_v(low_res_guidance)

                    low_res_guidance_query = low_res_guidance_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    low_res_guidance_key = low_res_guidance_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    low_res_guidance_value = low_res_guidance_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    low_res_guidance_query = attn.norm_q(low_res_guidance_query)
                if attn.norm_k is not None:
                    low_res_guidance_key = attn.norm_k(low_res_guidance_key)

                # Ensure good kernel behavior.
                low_res_guidance_query = low_res_guidance_query.contiguous()
                low_res_guidance_key = low_res_guidance_key.contiguous()
                low_res_guidance_value = low_res_guidance_value.contiguous()

                setattr(
                    attn,
                    "_lr_cond_qkv_cache",
                    {"key": cache_key, "value": (low_res_guidance_query, low_res_guidance_key, low_res_guidance_value)},
                )
        else:
            with enable_lora((attn.to_q, attn.to_k, attn.to_v), True):
                low_res_guidance_query = attn.to_q(low_res_guidance)
                low_res_guidance_key = attn.to_k(low_res_guidance)
                low_res_guidance_value = attn.to_v(low_res_guidance)

                low_res_guidance_query = low_res_guidance_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                low_res_guidance_key = low_res_guidance_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                low_res_guidance_value = low_res_guidance_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                low_res_guidance_query = attn.norm_q(low_res_guidance_query)
            if attn.norm_k is not None:
                low_res_guidance_key = attn.norm_k(low_res_guidance_key)
        query = torch.cat([query, low_res_guidance_query], dim=2)
        key = torch.cat([key, low_res_guidance_key], dim=2)
        value = torch.cat([value, low_res_guidance_value], dim=2)

    # =============================== Build Unified KV and Mask ===============================
    # Dense attention mask is disabled (standard attention path removed)
    attn_mask = None
    
    # Apply position embeddings to queries
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    # =============================== Perform Parallel Attention ===============================
    if low_res_guidance is not None:
        if SAGE_BLOCKSPARSE_AVAILABLE and model_config.get("use_sage_blocksparse", False):
            nheads = query.shape[1]
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
        
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scale,
        )
    
    # Reshape output
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    # Apply output projection
    if encoder_hidden_states is not None:
        # Split text and image outputs
        encoder_output = hidden_states[:, :text_seq_len, :]
        if low_res_guidance is not None:
            low_res_hidden_states = hidden_states[:, -low_res_guidance_key.shape[2]:, :]
            hidden_states_output = hidden_states[:, text_seq_len:-low_res_guidance_key.shape[2], :]
        else:
            hidden_states_output = hidden_states[:, text_seq_len:, :]
        
        # Apply projections

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
        # For single block, projection is handled by the block itself.
        if low_res_guidance is not None:
            hidden_states_output = hidden_states[:, :-low_res_guidance_key.shape[2], :]
            low_res_hidden_states = hidden_states[:, -low_res_guidance_key.shape[2]:, :]
            return hidden_states_output, low_res_hidden_states
        else:
            return hidden_states, None

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Optional[torch.FloatTensor],
    # Newly added codes
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = None,
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

# Keep block_forward and single_block_forward unchanged as they just call attn_forward
def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    low_res_temb: Optional[torch.FloatTensor],
    image_rotary_emb: Optional[torch.FloatTensor],
    # Newly added codes
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = None,
    block_idx: int=0,
    timestep: Optional[int] = None,
):
    # If low_res_temb is not present, disable all LoRA
    if low_res_temb is None:
        with enable_lora((self,), False):
            return _block_forward_impl(self, hidden_states, encoder_hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep)
    else:
        return _block_forward_impl(self, hidden_states, encoder_hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep)

def _block_forward_impl(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    low_res_temb: Optional[torch.FloatTensor],
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = None,
    block_idx: int=0,
    timestep: Optional[int] = None,
):
    if model_config is None:
        model_config = {}
    # hi-res tokens need to be finetuned
    with enable_lora((self.norm1.linear, self.norm1_context.linear), False):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
    if low_res_guidance is not None:
        # Use low_res_temb if available, otherwise fallback to temb
        embedding_for_lr = low_res_temb if low_res_temb is not None else temb
        lr_as_condition = bool(model_config.get("low_res_guidance_as_condition", False))
        use_lr_norm_cache = lr_as_condition and bool(model_config.get("cache_low_res_guidance_qkv", False))
        if use_lr_norm_cache:
            cache_key = (
                id(low_res_guidance),
                tuple(low_res_guidance.shape),
                str(low_res_guidance.dtype),
                str(low_res_guidance.device),
                id(embedding_for_lr),
                tuple(embedding_for_lr.shape),
                str(embedding_for_lr.dtype),
                str(embedding_for_lr.device),
            )
            cached = getattr(self, "_lr_cond_norm_cache", None)
            if cached is not None and cached.get("key") == cache_key:
                norm_low_res_guidance = cached["value"]
            else:
                with enable_lora((self.norm1.linear,), True):
                    norm_low_res_guidance, _, _, _, _ = self.norm1(low_res_guidance, emb=embedding_for_lr)
                # Keep it contiguous so downstream qkv caching is reliable/perf-friendly.
                norm_low_res_guidance = norm_low_res_guidance.contiguous()
                setattr(self, "_lr_cond_norm_cache", {"key": cache_key, "value": norm_low_res_guidance})
        else:
            with enable_lora((self.norm1.linear,), True):
                norm_low_res_guidance, lr_gate_msa, lr_shift_mlp, lr_scale_mlp, lr_gate_mlp = self.norm1(
                    low_res_guidance, emb=embedding_for_lr
                )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    # Attention - now using shifted window attention
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
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    if low_res_guidance is not None:
        if not bool(model_config.get("low_res_guidance_as_condition", False)):
            low_res_attn_output = lr_gate_msa.unsqueeze(1) * low_res_attn_output
            low_res_guidance = low_res_guidance + low_res_attn_output
    
    if context_attn_output is not None:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

    # LayerNorm + MLP
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    if low_res_guidance is not None:
        if not bool(model_config.get("low_res_guidance_as_condition", False)):
            norm_low_res_guidance = self.norm2(low_res_guidance)
            norm_low_res_guidance = (
                norm_low_res_guidance * (1 + lr_scale_mlp[:, None]) + lr_shift_mlp[:, None]
            )
       
    
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )

    # Feed-forward
    # hi-res tokens need to be finetuned
    with enable_lora((self.ff.net[2],), False):
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    if low_res_guidance is not None:
        if not bool(model_config.get("low_res_guidance_as_condition", False)):
            with enable_lora((self.ff.net[2],), True):
                lr_ff_output = self.ff(norm_low_res_guidance)
                lr_ff_output = lr_gate_mlp.unsqueeze(1) * lr_ff_output
    
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output

    # Process outputs
    hidden_states = hidden_states + ff_output
    if low_res_guidance is not None:
        if not bool(model_config.get("low_res_guidance_as_condition", False)):
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
    model_config: Optional[Dict[str, Any]] = None,
    block_idx: int=0,
    timestep: Optional[int] = None,
):
    # If low_res_temb is not present, disable all LoRA
    if low_res_temb is None:
        with enable_lora((self,), False):
            return _single_block_forward_impl(self, hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep)
    else:
        return _single_block_forward_impl(self, hidden_states, temb, low_res_temb, image_rotary_emb, low_res_guidance, model_config, block_idx, timestep)

def _single_block_forward_impl(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    low_res_temb: Optional[torch.FloatTensor],
    image_rotary_emb: Optional[torch.FloatTensor],
    low_res_guidance: Optional[torch.Tensor],
    model_config: Optional[Dict[str, Any]] = None,
    block_idx: int=0,
    timestep: Optional[int] = None,
):
    if model_config is None:
        model_config = {}
    residual = hidden_states

    # Process high-res tokens (without LoRA for norm and proj_mlp)
    with enable_lora((self.norm.linear, self.proj_mlp), False):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    # Process low-res guidance tokens (without LoRA for norm and proj_mlp)
    if low_res_guidance is not None:
        low_res_residual = low_res_guidance
        # Use low_res_temb if available, otherwise fallback to temb
        embedding_for_lr = low_res_temb if low_res_temb is not None else temb
        with enable_lora((self.norm.linear, self.proj_mlp), True):
            norm_low_res_guidance, lr_gate = self.norm(low_res_guidance, emb=embedding_for_lr)
            lr_mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_low_res_guidance))

    # Attention processing using shifted window attention
    attn_forward_output = attn_forward(
        self.attn,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        low_res_guidance=None if low_res_guidance is None else norm_low_res_guidance,
        model_config=model_config,
        block_idx=block_idx,
        timestep=timestep,
    )
    
    if low_res_guidance is None:
        attn_output, _ = attn_forward_output
    else:
        attn_output, low_res_attn_output = attn_forward_output

    # Output projection for high-res tokens (without LoRA)
    with enable_lora((self.proj_out,), False):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    
    # Output projection for low-res guidance tokens (with LoRA)
    if low_res_guidance is not None:
        with enable_lora((self.proj_out,), True):
            low_res_guidance = torch.cat([low_res_attn_output, lr_mlp_hidden_states], dim=2)
            lr_gate = lr_gate.unsqueeze(1)
            low_res_guidance = lr_gate * self.proj_out(low_res_guidance)
            low_res_guidance = low_res_residual + low_res_guidance

    # Clip values if using fp16
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
        if low_res_guidance is not None:
            low_res_guidance = low_res_guidance.clip(-65504, 65504)

    return (hidden_states, low_res_guidance) if low_res_guidance is not None else (hidden_states, None)