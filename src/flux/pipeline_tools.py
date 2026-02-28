import torch
from torch import Tensor
from diffusers.utils import logging
from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import logger as logger2
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from torchvision.transforms import GaussianBlur
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.models.embeddings import get_1d_rotary_pos_embed


def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    # Align inputs with VAE weights to avoid CPU/CUDA dtype mismatches
    images = images.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids

def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids

def prepare_text_input2(
    pipeline: Flux2KleinPipeline,
    prompts,
    max_sequence_length=512,
    prompt_embeds=None,
    text_encoder_out_layers=(9, 18, 27),
):
    logger2.setLevel(logging.ERROR)
    device = getattr(pipeline, "device", None) or getattr(pipeline, "_execution_device", None)
    (
        prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_embeds=prompt_embeds,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    )
    logger2.setLevel(logging.WARNING)
    return prompt_embeds, text_ids

from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_latents
def encode_images_tiled2(pipeline: Flux2KleinPipeline, image: Tensor):
    """
    Encode a batch of images to FLUX2 latent tokens + ids.

    This mirrors the upstream `Flux2KleinPipeline._encode_vae_image` logic
    (VAE encode -> patchify -> BN normalize), then packs to token form and
    constructs per-token position ids.
    """
    if image.ndim != 4:
        image = image.unsqueeze(0)

    # `_encode_vae_image` expects its input on the VAE's device/dtype (see upstream prepare_image_latents).
    image = image.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)

    # Use tiling for high-res safety; restore original setting after.
    pipeline.vae.enable_tiling()
    try:
        # Returns patchified+BN-normalized latents: (B, 128, H/16, W/16) for 1024x1024 inputs.
        generator = torch.Generator(device=pipeline.vae.device)
        image_latents = pipeline._encode_vae_image(image=image, generator=generator)
    finally:
        pipeline.vae.disable_tiling()

    # Build ids on the patchified latent grid (matches packed token layout).
    ids = pipeline._prepare_latent_ids(image_latents).to(device=pipeline.device)

    # Pack to token form for the transformer: (B, H*W, C)
    tokens = pipeline._pack_latents(image_latents).to(device=pipeline.device, dtype=pipeline.dtype)

    return tokens, ids



def encode_images_tiled(pipeline: FluxPipeline, images: Tensor):
    """
    Encodes images using VAE tiling to support high-resolution inputs.
    """
    images = pipeline.image_processor.preprocess(images)
    # Ensure tiled encode runs on same device/dtype as VAE weights
    images = images.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)

    pipeline.vae.enable_tiling()
    latents = pipeline.vae.encode(images).latent_dist.sample()
    pipeline.vae.disable_tiling()

    latents = (
        latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor

    images_tokens = pipeline._pack_latents(latents, *latents.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2],
        latents.shape[3],
        pipeline.device,
        pipeline.dtype,
    )

    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids
    
def FluxPosEmbedForward(
    pos_embed_func,
    ids: torch.Tensor,
    ntk_factor: float=1.0,
    theta: float=10000.0,
    linear_factor: int=1,
):
    n_axes = ids.shape[-1]
    cos_out = []
    sin_out = []
    pos = ids.float()
    is_mps = ids.device.type == "mps"
    freqs_dtype = torch.float32 if is_mps else torch.float64
    for i in range(n_axes):
        cos, sin = get_1d_rotary_pos_embed(
            pos_embed_func.axes_dim[i],
            pos[:, i],
            repeat_interleave_real=True,
            use_real=True,
            freqs_dtype=freqs_dtype,
            ntk_factor=ntk_factor,
            linear_factor = linear_factor,
            theta=theta,
        )
        cos_out.append(cos)
        sin_out.append(sin)
    freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
    freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
    return freqs_cos, freqs_sin        

def Flux2PosEmbedForward(
    pos_embed_func,
    ids: torch.Tensor,
    ntk_factor: float=1.0,
    theta: float=2000.0,
    linear_factor: int=1,
):
    # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        # Unlike Flux 1, loop over len(self.axes_dim) rather than ids.shape[-1]
        for i in range(len(pos_embed_func.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                pos_embed_func.axes_dim[i],
                pos[..., i],
                ntk_factor=ntk_factor,
                linear_factor=linear_factor,
                theta=theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin

def create_butterworth_filter(shape, cutoff=0.3, order=2, device=torch.device("cpu"), dtype=torch.bfloat16):
    """
    Create a Butterworth filter for frequency filtering.
    """
    B, C, H, W = shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    x = x.to(device).to(dtype)
    y = y.to(device).to(dtype)
    center_y, center_x = H // 2, W // 2
    d = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).to(device).to(dtype)
    max_d = min(H, W) // 2
    normalized_d = d / max_d
    filter = 1 / (1 + (normalized_d / cutoff) ** (2 * order)).to(device).to(dtype)
    return filter

def apply_freq_filter(x, filter_mask, low_pass=True, method="fft"):
    """
    Apply frequency filtering to a tensor using FFT/DWT."""
    if method == "fft":
        # Store original dtype and convert to float32 for FFT operations
        original_dtype = x.dtype
        x_float32 = x.float()
        filter_mask_float32 = filter_mask.float()
        
        x_freq = torch.fft.fft2(x_float32)
        x_freq_shifted = torch.fft.fftshift(x_freq)

        if low_pass:
            filtered_freq = x_freq_shifted * filter_mask_float32
        else:
            filtered_freq = x_freq_shifted * (1 - filter_mask_float32)

        filtered_freq_shifted = torch.fft.ifftshift(filtered_freq)
        filtered_freq = torch.fft.ifft2(filtered_freq_shifted)
        
        # Convert back to original dtype
        return filtered_freq.real.to(original_dtype)
    else:
        raise ValueError(f"Invalid method: {method}, not implemented.")

def save_image(self, latents, height, width, output_path, output_type, latent_ids=None):
    if latent_ids is None:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        image[0].save(output_path)
        return image[0]
    else:
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        image[0].save(output_path)
        return image[0]

def decode_vae_latents_tiled(self, latents, height, width, output_type, latent_ids=None):
    if latent_ids is None:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
    else:
        latents = self._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)
    return image

def encode_vae_latents(self, image, batch_size, num_channels_latents, height, width,):
    height = 2 * (int(height) // self.vae_scale_factor)
    width = 2 * (int(width) // self.vae_scale_factor)
    latents = self.vae.encode(image.to(self.vae.dtype).to(self.vae.device)).latent_dist.mode()
    latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
    latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
    latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, self.vae.device, self.vae.dtype)
    return latents, latent_image_ids



def window_permutation(H: int, W: int, Wh: int, Ww: int, hierarchical: bool = False) -> torch.Tensor:
    """
    Return a permutation idx such that x.view(H*W)[idx] is reordered
    window-by-window (raster scan over windows, raster scan inside each window).
    If hierarchical=True, first permute to window-first order with Wh x Ww windows,
    then within each window, permute to subwindow-first order with (Wh/4) x (Ww/4) subwindows.
    """
    if not (H % Wh == 0 and W % Ww == 0):
        raise ValueError(f"H({H}), W({W}) must be divisible by Wh({Wh}), Ww({Ww})")
    
    idx = torch.arange(H * W).reshape(H, W)
    
    if not hierarchical:
        # Non-hierarchical permutation
        patches = idx.unfold(0, Wh, Wh).unfold(1, Ww, Ww)  # (nWh, nWw, Wh, Ww)
        perm = patches.contiguous().view(-1)               # (H*W,)
        return perm
    
    # Hierarchical permutation
    if not (Wh % 4 == 0 and Ww % 4 == 0):
        raise ValueError(f"For hierarchical=True, Wh({Wh}) and Ww({Ww}) must be divisible by 4")
    
    # First level: permute to window-first order with Wh x Ww windows
    patches = idx.unfold(0, Wh, Wh).unfold(1, Ww, Ww)  # (nWh, nWw, Wh, Ww)
    
    # Second level: within each window, permute to subwindow-first order
    sub_Wh, sub_Ww = Wh // 4, Ww // 4
    # Shape: (nWh, nWw, 4, 4, sub_Wh, sub_Ww)
    hierarchical_patches = patches.unfold(2, sub_Wh, sub_Wh).unfold(3, sub_Ww, sub_Ww)
    
    # Reshape to get subwindow-first order. The memory layout from unfold is already
    # correct for a raster-scan-style flattening, so no permute is needed.
    perm = hierarchical_patches.contiguous().view(-1)   # (H*W,)
    return perm

def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv

def hilbert_window_permutation(H: int, W: int, Wh: int, Ww: int) -> torch.Tensor:
    """
    Return a permutation idx that orders tokens using Hilbert curves within windows.
    First, tokens are grouped into windows of size Wh x Ww.
    Within each window, tokens are ordered following a Hilbert curve pattern.
    Windows themselves are ordered in raster scan order.
    
    Args:
        H: Height of the 2D grid
        W: Width of the 2D grid  
        Wh: Window height
        Ww: Window width
    
    Returns:
        Permutation tensor of shape (H*W,) containing indices in Hilbert-windowed order
    """
    if not (H % Wh == 0 and W % Ww == 0):
        raise ValueError(f"H({H}), W({W}) must be divisible by Wh({Wh}), Ww({Ww})")
    
    def hilbert_2d_to_index(n, i, j):
        """Convert (i,j) coordinates to Hilbert curve index for n x n grid"""
        if n <= 1:
            return 0
        
        half = n // 2
        t = 0
        
        if i < half:
            if j < half:
                # Bottom-left quadrant
                t += hilbert_2d_to_index(half, j, i)
            else:
                # Top-left quadrant  
                t += 2 * half * half + hilbert_2d_to_index(half, i, j - half)
        else:
            if j < half:
                # Bottom-right quadrant
                t += 3 * half * half + hilbert_2d_to_index(half, half - 1 - j, half - 1 - (i - half))
            else:
                # Top-right quadrant
                t += half * half + hilbert_2d_to_index(half, i - half, j - half)
                
        return t
    
    def hilbert_index_to_2d(n, idx):
        """Convert Hilbert curve index to (i,j) coordinates for n x n grid"""
        i, j = 0, 0
        t = idx
        s = 1
        
        while s < n:
            rx = 1 & (t >> 1)
            ry = 1 & (t ^ rx)
            
            if ry == 0:
                if rx == 1:
                    i = s - 1 - i
                    j = s - 1 - j
                i, j = j, i
                
            i += s * rx
            j += s * ry
            t >>= 2
            s *= 2
            
        return i, j
    
    # Find the next power of 2 that accommodates the window size
    window_size = max(Wh, Ww)
    n = 1
    while n < window_size:
        n *= 2
    
    # Create original index grid
    idx = torch.arange(H * W).reshape(H, W)
    
    # Split into windows
    num_windows_h = H // Wh
    num_windows_w = W // Ww
    
    perm_list = []
    
    # Process each window
    for win_i in range(num_windows_h):
        for win_j in range(num_windows_w):
            # Extract window indices
            start_h = win_i * Wh
            end_h = start_h + Wh
            start_w = win_j * Ww
            end_w = start_w + Ww
            
            window_indices = idx[start_h:end_h, start_w:end_w]
            
            # Create Hilbert ordering within this window
            hilbert_coords = []
            original_coords = []
            
            for h in range(Wh):
                for w in range(Ww):
                    if h < n and w < n:  # Only process coordinates within n x n
                        hilbert_idx = hilbert_2d_to_index(n, h, w)
                        hilbert_coords.append(hilbert_idx)
                        original_coords.append((h, w))
            
            # Sort by Hilbert index and extract the corresponding window indices
            sorted_pairs = sorted(zip(hilbert_coords, original_coords))
            
            for _, (h, w) in sorted_pairs:
                if h < Wh and w < Ww:  # Ensure we're within window bounds
                    perm_list.append(window_indices[h, w].item())
    
    return torch.tensor(perm_list, dtype=torch.long)

def gaussian_blur_image_sharpening(image, kernel_size=3, sigma=(0.1, 2.0), alpha=1):
    gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image_blurred = gaussian_blur(image)
    image_sharpened = (alpha + 1) * image - alpha * image_blurred

    return image_sharpened