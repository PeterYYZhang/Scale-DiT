import torch
import yaml, os
import time
import csv
import torch.nn.functional as F
import pywt
from PIL import Image
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from .transformer2 import tranformer_forward
from typing import List, Union, Optional, Dict, Any, Callable
from .pipeline_tools import (
    save_image,
    window_permutation,
    inverse_permutation,
    encode_images_tiled2,
)

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2PipelineOutput,
    compute_empirical_mu,
    retrieve_timesteps,
    np,
)

def _env_flag(name: str, default: Optional[bool] = None) -> Optional[bool]:
    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    if v == "auto":
        return None
    return default


def _move_transformer_blocks_to_device(transformer, device: torch.device) -> None:
    """
    Ensure all transformer blocks are resident on `device`.
    Needed if we used sequential CPU offload during sampling but will resume training.
    """
    for name in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, name, None)
        if blocks is None:
            continue
        for b in blocks:
            try:
                b.to(device)
            except Exception:
                pass


def _is_8192x8192(model_config: Dict[str, Any]) -> bool:
    cfg = model_config or {}
    sz = cfg.get("image_size", None)
    try:
        h = int(sz[0])
        w = int(sz[1])
        return h == 8192 and w == 8192
    except Exception:
        return False


def split_frequency_components_dwt(x, wavelet='haar', level=1):
    device = x.device
    dtype = x.dtype
    x = x.type(torch.float32)
    x = x.cpu().numpy()

    B, C, H, W = x.shape
    low_freq_components = []

    # Using list comprehension to improve performance
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec2(x[b, c], wavelet=wavelet, level=level)
            low_freq, *high_freq = coeffs
            low_freq_components.append([low_freq] + [(np.zeros_like(detail[0]), np.zeros_like(detail[1]), np.zeros_like(detail[2])) for detail in high_freq])

    # Convert list of numpy arrays to a single numpy array for better performance
    x_low_freq = np.stack([pywt.waverec2(low_freq_components[i], wavelet=wavelet) for i in range(B * C)])

    # Convert the numpy array to a tensor
    x_low_freq = torch.from_numpy(x_low_freq).view(B, C, H, W).type(dtype).to(device)

    return x_low_freq


def get_config(config_path: str = None):
    config_path = config_path or os.environ.get("XFL_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Newly added codes
    low_res_latents: Optional[torch.FloatTensor] = None,
    hr_inference_steps: Optional[int] = 14,
    hr_guidance_scale: Optional[float] = 4.5,
    save_path: Optional[str] = None,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
        # Newly added codes
        low_res_latents,
        hr_inference_steps,
        hr_guidance_scale,
        save_path,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def _as_int_timestep(t) -> int:
    try:
        return int(t.item())
    except Exception:
        try:
            return int(t)
        except Exception:
            return -1


def _cuda_device_index(device: torch.device) -> Optional[int]:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def _cuda_mem_snapshot_bytes(device_index: int) -> Dict[str, float]:
    """
    Returns a snapshot of CUDA memory stats in bytes.
    - free/total come from the CUDA driver (close to what `nvidia-smi` reports; used = total - free)
    - allocated/reserved/peaks come from PyTorch's caching allocator
    """
    torch.cuda.synchronize(device_index)
    free, total = torch.cuda.mem_get_info(device_index)
    return {
        "free": float(free),
        "total": float(total),
        "allocated": float(torch.cuda.memory_allocated(device_index)),
        "reserved": float(torch.cuda.memory_reserved(device_index)),
        "max_allocated": float(torch.cuda.max_memory_allocated(device_index)),
        "max_reserved": float(torch.cuda.max_memory_reserved(device_index)),
    }


@torch.no_grad()
def generate(
    pipeline: Flux2KleinPipeline,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})
    seed_everything(model_config.get("seed", 42))
    self = pipeline
    # Force-disable scheduler dynamic shifting for inference and pin time shift to 10.
    # (Some configs enable dynamic shifting based on image token count; we want a fixed shift.)
    if hasattr(self, "scheduler") and hasattr(self.scheduler, "config"):
        if hasattr(self.scheduler.config, "use_dynamic_shifting"):
            self.scheduler.config.use_dynamic_shifting = False
        if hasattr(self.scheduler.config, "time_shift"):
            self.scheduler.config.time_shift = 10
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
        # Newly added codes
        low_res_latents,
        hr_inference_steps,
        hr_guidance_scale,
        save_path,
    ) = prepare_params(**params)
    negative_prompt = "low quality, bad quality, sketches"
    self.vae.enable_tiling()
    guidance_scale = model_config.get("lr_guidance_scale", guidance_scale)
    
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    original_image_size = model_config.get("image_size", (height, width))
    model_config["image_size"] = (height, width)

    os.makedirs(save_path, exist_ok=True)
    
    # 1. Check inputs. Raise error if not correct
    # Klein signature: (prompt, height, width, prompt_embeds=None, callback_on_step_end_tensor_inputs=None, guidance_scale=None)
    lr_height, lr_width = model_config.get("joint_denoise_size", [256, 256])
    self.check_inputs(
        prompt=prompt,
        height=lr_height,
        width=lr_width,
        prompt_embeds=prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        guidance_scale=guidance_scale,
    )

    self._guidance_scale = guidance_scale
    # Klein pipeline uses `attention_kwargs`; keep accepting `joint_attention_kwargs` for backward compatibility.
    self._attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Enable sequential CPU offload of transformer blocks during no_grad sampling.
    # Priority:
    #   - explicit `model_config["cpu_offload_transformer_blocks"]`
    #   - env `XFL_CPU_OFFLOAD_TRANSFORMER_BLOCKS` ("1"/"0"/"auto")
    #   - auto for very large images on CUDA
    env_offload = _env_flag("XFL_CPU_OFFLOAD_TRANSFORMER_BLOCKS", default=None)
    if "cpu_offload_transformer_blocks" not in model_config and env_offload is not None:
        model_config["cpu_offload_transformer_blocks"] = bool(env_offload)
    if "cpu_offload_transformer_blocks" not in model_config:
        # Auto: only for CUDA, only for 8192x8192 (avoid slowing down smaller resolutions).
        model_config["cpu_offload_transformer_blocks"] = bool(device.type == "cuda" and _is_8192x8192(model_config))

    # Gate: by default, only allow this offload path for exactly 8192x8192.
    # You can override by setting `cpu_offload_transformer_blocks_only_8192: false`.
    if bool(model_config.get("cpu_offload_transformer_blocks_only_8192", True)) and not _is_8192x8192(model_config):
        model_config["cpu_offload_transformer_blocks"] = False
    # A moderate default; can override via YAML or env by setting this key in model_config.
    model_config.setdefault("cpu_offload_empty_cache_every", int(os.environ.get("XFL_CPU_OFFLOAD_EMPTY_CACHE_EVERY", "4")))

    # Ensure text encoder is on the correct device before encoding
    if hasattr(self, "text_encoder") and self.text_encoder is not None:
        self.text_encoder.to(device)
    if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
        self.text_encoder_2.to(device)

    # Flux2 encodes a single prompt stream. If a secondary prompt is provided (legacy Flux API),
    # concatenate it for best-effort compatibility.
    effective_prompt = prompt
    if prompt_embeds is None and prompt_2 is not None:
        if effective_prompt is None:
            effective_prompt = prompt_2
        elif isinstance(effective_prompt, str) and isinstance(prompt_2, str):
            effective_prompt = effective_prompt + "\n" + prompt_2
        elif isinstance(effective_prompt, list) and isinstance(prompt_2, str):
            effective_prompt = [p + "\n" + prompt_2 for p in effective_prompt]
        elif isinstance(effective_prompt, str) and isinstance(prompt_2, list):
            effective_prompt = [effective_prompt + "\n" + p2 for p2 in prompt_2]
        elif isinstance(effective_prompt, list) and isinstance(prompt_2, list) and len(effective_prompt) == len(prompt_2):
            effective_prompt = [p + "\n" + p2 for p, p2 in zip(effective_prompt, prompt_2)]
        else:
            raise ValueError(
                f"Unsupported prompt/prompt_2 types: {type(effective_prompt)} and {type(prompt_2)}"
            )

    text_encoder_out_layers = tuple(model_config.get("text_encoder_out_layers", (9, 18, 27)))
    prompt_embeds, text_ids = self.encode_prompt(
        prompt=effective_prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        prompt_embeds=prompt_embeds,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    )
    pooled_prompt_embeds = None
    if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                # prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

    # Offload the encoder to CPU after encoding
    if hasattr(self, "text_encoder") and self.text_encoder is not None:
        self.text_encoder.to("cpu")
    if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
        self.text_encoder_2.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_latents_channels=num_channels_latents,
        height=model_config.get("joint_denoise_size", [256, 256])[0],
        width=model_config.get("joint_denoise_size", [256, 256])[1],
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=latents,
    )
    low_res_latents_ids = latent_image_ids
    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    
    # 6. Denoising loop for low-res image
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = tranformer_forward(
                    self.transformer,
                    model_config=model_config,
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=None,
                    pooled_projections=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]
            if self.do_classifier_free_guidance:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = tranformer_forward(
                    self.transformer,
                    model_config=model_config,
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=None,
                    pooled_projections=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            # original_pred_x0 = latents - noise_pred * self.scheduler.sigmas[i]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    # load the low-res image using center crop
    # low_res_path = "/scratch/yuyao/Scale-DiT/c27a51768b26f8a2baafc8d785fad46f 2.jpg"
    low_res_path = None
    if low_res_path is not None: #TODO: need to fix I hate new diffusers version
        joint_size = model_config.get("joint_denoise_size", [256, 256])
        image = Image.open(low_res_path).convert("RGB")
        image = image.resize((joint_size[1], joint_size[0]), resample=Image.BICUBIC)
        image = self.image_processor.preprocess(image)
        image = image.to(device=self.device, dtype=self.dtype)

        latents, latent_image_ids = encode_images_tiled2(self, image)
        latents = latents.to(device)
        latent_image_ids = latent_image_ids.to(device)

    # Save low-res image
    # Keep LR tokens for joint denoise guidance (no need to clone; we rebind `latents` later).
    lr_latents_copy = latents
    low_res_pil = save_image(
        self,
        latents,
        model_config.get("joint_denoise_size", [256, 256])[0],
        model_config.get("joint_denoise_size", [256, 256])[1],
        os.path.join(save_path, f"low_res_image_{prompt[:50]}.png"),
        output_type="pil",
        latent_ids=latent_image_ids,
    )
    
    # Configure model to use BlockSparseAttention for high-res generation
    if model_config.get("use_block_sparse_attention", False):
        model_config["use_flash_attention"] = False 
        model_config["use_block_sparse_attention"] = True
    # 7. Denoising loop for high-res image
    # Build HR guidance latents efficiently (avoid VAE decode -> resize -> VAE encode).
    # We upsample *in latent space* then repack to token form.
    height, width = model_config.get("image_size", (height, width))[0], model_config.get("image_size", (height, width))[1]
    patch_size = int(model_config.get("patch_size", 16))
    joint_h = model_config.get("joint_denoise_size", [256, 256])[0]
    joint_w = model_config.get("joint_denoise_size", [256, 256])[1]

    # lr_latents_spatial = self._unpack_latents(latents, joint_h, joint_w, self.vae_scale_factor)
    lr_latents_spatial = self._unpack_latents_with_ids(latents, latent_image_ids)
    # Target latent spatial size used by FLUX packing.
    # For FLUX2, token grid is on the patchified latent map (image_size / patch_size).
    pack_h = height // patch_size
    pack_w = width // patch_size
    hr_latents_spatial = F.interpolate(lr_latents_spatial, size=(pack_h, pack_w), mode="bicubic", align_corners=False)
    latents_guidance = self._pack_latents(
        hr_latents_spatial,
    ).to(device=device)
    # For HR generation, we want `latents` / `latent_image_ids` to reflect the HR token layout.
    # If `interpolation_init` is enabled, we will start from these guidance latents; otherwise
    # `prepare_latents` below will overwrite `latents` with random noise latents.
    latents = latents_guidance

    # Prepare HR latent ids (same logic as `encode_images_tiled`, but without re-encoding).
    latent_image_ids = self._prepare_latent_ids(
        latents.reshape(batch_size, -1, pack_h, pack_w),
        # hr_latents_spatial.shape[0],
        # hr_latents_spatial.shape[2],
        # hr_latents_spatial.shape[3],
        # device,
        # self.dtype,
    )
    latent_image_ids = latent_image_ids.to(device)

    # For output/debugging only: an upsampled LR image on CPU (no extra GPU work).
    lr_guidance_img_save = None
    if low_res_pil is not None:
        lr_guidance_img_save = low_res_pil.resize((width, height), resample=Image.BICUBIC)

    # Optionally offload VAE during the HR denoising loop to free GPU memory.
    if model_config.get("offload_vae_during_hr", True):
        try:
            self.vae.to("cpu")
            if getattr(self, "vae2", None) is not None:
                self.vae2.to("cpu")
        except Exception:
            pass

    # Drop large intermediate tensors as soon as possible.
    del lr_latents_spatial, hr_latents_spatial
    image_seq_len = latents.shape[1]
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    if model_config.get("hr_guidance_scale", None) is not None:
        guidance = torch.tensor([model_config.get("hr_guidance_scale", None)], device=device)
        guidance = guidance.expand(latents.shape[0])

    dlfg_timesteps = self.scheduler.timesteps[-hr_inference_steps:]
    
    if model_config.get("interpolation_init", False):
        noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
        latents = self.scheduler.scale_noise(latents, dlfg_timesteps[None, 0], noise,).to(self.transformer.dtype)
        # noise, _ = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     model_config.get("image_size", [1024, 1024])[0],
        #     model_config.get("image_size", [1024, 1024])[1],
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        # )
        # alpha = model_config.get("interpolation_alpha", 0.5)
        # latents = (1 - dlfg_timesteps[0]/1000) * latents + dlfg_timesteps[0]/1000 * noise
    else:
        latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        model_config.get("image_size", [1024, 1024])[0],
        model_config.get("image_size", [1024, 1024])[1],
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )

    hr_permutation_idx = window_permutation(
        H=model_config.get("image_size", [1024, 1024])[0] // patch_size,
        W=model_config.get("image_size", [1024, 1024])[1] // patch_size,
        Wh=16,
        Ww=16,
    )
    inv_hr_permutation_idx = inverse_permutation(hr_permutation_idx)
    # import pdb; pdb.set_trace()
    latent_image_ids = latent_image_ids[:, hr_permutation_idx, ]
    latents = latents[:, hr_permutation_idx, ]
    # Align guidance tokens to the same permutation order used for HR denoising.
    latents_guidance = latents_guidance[:, hr_permutation_idx, ].to(dtype=latents.dtype, device=device)
    if model_config.get("joint_denoise_size", [256, 256])[0] > 256:
        lr_permutation_idx = window_permutation(
            H=model_config.get("joint_denoise_size", [256, 256])[0] // patch_size,
            W=model_config.get("joint_denoise_size", [256, 256])[1] // patch_size,
            Wh=4,
            Ww=4,
        )
        lr_latents_copy = lr_latents_copy[:, lr_permutation_idx, ]
        low_res_latents_ids = low_res_latents_ids[:, lr_permutation_idx, ]
    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.tensor([hr_guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None 

    # Projected-flow LR guidance configuration (cosine_shift).
    guidance_schedule = model_config.get("guidance_schedule", "disable")
    dwt_level = int(model_config.get("dwt_level", 1))
    dwt_wavelet = model_config.get("dwt_wavelet", "haar")
    pack_h = height // patch_size
    pack_w = width // patch_size

    def _lowpass_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D) in *permuted* HR order.
        Returns: low-frequency tokens (B, N, D) in the same permuted order, computed via DWT.
        """
        spatial = self._unpack_latents_with_ids(tokens, latent_image_ids)
        spatial_low = split_frequency_components_dwt(spatial, wavelet=dwt_wavelet, level=dwt_level)
        tokens_low = self._pack_latents(spatial_low)
        return tokens_low[:, hr_permutation_idx, :]

    # HR denoising loop memory tracking (starts AFTER LR interpolation/upsampling above).
    # We reset peak stats here (right before the HR loop) so the measurements exclude
    # any transient extra memory from interpolation.
    hr_mem_enabled = bool(model_config.get("log_hr_memory", save_path is not None))
    hr_mem_every = int(model_config.get("hr_mem_log_every", 1))
    hr_mem_path = os.path.join(save_path, "hr_memory.csv") if (hr_mem_enabled and save_path) else None
    hr_mem_device_index = _cuda_device_index(device)
    hr_mem_t0 = None
    hr_mem_base = None
    hr_mem_writer = None
    hr_mem_fh = None

    if hr_mem_path is not None and hr_mem_device_index is not None:
        os.makedirs(save_path, exist_ok=True)
        torch.cuda.reset_peak_memory_stats(hr_mem_device_index)
        hr_mem_t0 = time.time()
        hr_mem_base = _cuda_mem_snapshot_bytes(hr_mem_device_index)
        hr_mem_fh = open(hr_mem_path, "w", newline="")
        hr_mem_writer = csv.DictWriter(
            hr_mem_fh,
            fieldnames=[
                "phase",
                "step",
                "timestep",
                "wall_time_s",
                "free",
                "total",
                "allocated",
                "reserved",
                "max_allocated",
                "max_reserved",
                "free_delta",
                "allocated_delta",
                "reserved_delta",
                "max_allocated_delta",
                "max_reserved_delta",
            ],
        )
        hr_mem_writer.writeheader()
        hr_mem_writer.writerow(
            {
                "phase": "hr_start",
                "step": -1,
                "timestep": -1,
                "wall_time_s": 0.0,
                **hr_mem_base,
                "free_delta": 0.0,
                "allocated_delta": 0.0,
                "reserved_delta": 0.0,
                "max_allocated_delta": 0.0,
                "max_reserved_delta": 0.0,
            }
        )
    torch.cuda.empty_cache()
    ## denoising loop starts here
    with self.progress_bar(total=hr_inference_steps) as progress_bar:
        for i, t in enumerate(dlfg_timesteps):
            if self.interrupt:
                continue

            # apply freq filtering
            if model_config.get("freq_filtering", False):
                raise NotImplementedError("Freq filtering is not implemented yet")
            
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            # Remove batch dim when 3D (deprecated); transformer expects 2D
            lr_guidance = lr_latents_copy[0] if lr_latents_copy.ndim == 3 else lr_latents_copy
            lr_ids = low_res_latents_ids[0] if low_res_latents_ids.ndim == 3 else low_res_latents_ids
            with self.transformer.cache_context("cond"):
                noise_pred = tranformer_forward(
                    self.transformer,
                    model_config=model_config,
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                    # Newly added codes
                    low_res_guidance=lr_guidance,
                    low_res_img_ids=lr_ids,
                    low_res_timestep=0,
                )[0]
            noise_pred = noise_pred[:, : latents.size(1) :]
            # if self.do_classifier_free_guidance:
            #     with self.transformer.cache_context("uncond"):
            #         neg_noise_pred = tranformer_forward(
            #         self.transformer,
            #         model_config=model_config,
            #         hidden_states=latents,
            #         timestep=timestep / 1000,
            #         guidance=None,
            #         pooled_projections=None,
            #         encoder_hidden_states=negative_prompt_embeds,
            #         txt_ids=negative_text_ids,
            #         img_ids=latent_image_ids,
            #         joint_attention_kwargs=self.attention_kwargs,
            #         return_dict=False,
            #         # Newly added codes
            #         low_res_guidance=lr_guidance,
            #         low_res_img_ids=lr_ids,
            #         low_res_timestep=0,
            #     )[0]
            #     neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
            #     noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
            # Apply projected-flow guidance (cosine_shift) using the upsampled LR guidance latents.
            if guidance_schedule != "disable":
                # Flow projection velocity that moves current sample toward the guidance latent.
                denom = (timestep / 1000.0).reshape(-1, 1, 1) + 1e-6
                fp_v = -(latents_guidance - latents) / denom

                # Cosine schedule: starts near 1, ends near 0 (matches upstream cosine_shift behavior).
                alpha = 0.5 * (
                    1.0
                    + torch.cos(
                        torch.pi
                        * (torch.tensor(float(i), device=device, dtype=latents.dtype) / torch.tensor(float(hr_inference_steps), device=device, dtype=latents.dtype))
                    )
                )

                if guidance_schedule == "cosine_decay":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + alpha * (fp_v_low - noise_pred_low)
                elif guidance_schedule == "cosine_hp_decay":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + alpha * (fp_v_low - noise_pred_low)
                elif guidance_schedule == "cosine_shift":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + alpha * (fp_v - noise_pred) + (1-alpha) * (fp_v_low - noise_pred_low)
                elif guidance_schedule == "constant":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + (fp_v_low - noise_pred_low)
                elif guidance_schedule == "constant_hp":
                    noise_pred = noise_pred + 0.25 * (fp_v - noise_pred)
                else:
                    raise ValueError(f"Unknown guidance_schedule={guidance_schedule}. Expected one of disable|cosine_decay|cosine_shift|constant.")

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # Optional per-step memory logging for the HR denoising loop.
            if hr_mem_writer is not None and (
                hr_mem_every <= 1 or (i % hr_mem_every == 0) or (i == len(dlfg_timesteps) - 1)
            ):
                snap = _cuda_mem_snapshot_bytes(hr_mem_device_index)
                hr_mem_writer.writerow(
                    {
                        "phase": "hr_step",
                        "step": int(i),
                        "timestep": _as_int_timestep(t),
                        "wall_time_s": float(time.time() - hr_mem_t0),
                        **snap,
                        "free_delta": float(snap["free"] - hr_mem_base["free"]),
                        "allocated_delta": float(snap["allocated"] - hr_mem_base["allocated"]),
                        "reserved_delta": float(snap["reserved"] - hr_mem_base["reserved"]),
                        "max_allocated_delta": float(snap["max_allocated"] - hr_mem_base["max_allocated"]),
                        "max_reserved_delta": float(snap["max_reserved"] - hr_mem_base["max_reserved"]),
                    }
                )

            # call the callback, if provided
            if i == len(dlfg_timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if hr_mem_fh is not None:
        hr_mem_fh.close()
    inv_hr_permutation_idx = inverse_permutation(hr_permutation_idx)
    latents = latents[:, inv_hr_permutation_idx, ]
    latent_image_ids = latent_image_ids[:, inv_hr_permutation_idx, ]
    self.vae.enable_tiling()
    self.vae.enable_slicing()
    # 8. Save high-res image
    if output_type == "latent":
        image = latents
    else:
        # If we offloaded VAE during HR loop, bring it back for decode.
        if model_config.get("offload_vae_during_hr", True):
            try:
                self.vae.to(device)
                if getattr(self, "vae2", None) is not None:
                    self.vae2.to(device)
            except Exception:
                pass
        # latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = self._unpack_latents_with_ids(latents, latent_image_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    
    # Offload all models
    self.maybe_free_model_hooks()
    model_config["image_size"] = original_image_size
    # put the text encoder back to the device
    if hasattr(self, 'text_encoder') and self.text_encoder is not None:
        self.text_encoder.to(device)
    if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
        self.text_encoder_2.to(device)

    # If we used sequential CPU offload during sampling inside a training run, restore blocks to GPU
    # so the next training forward doesn't hit device-mismatch errors.
    if bool(model_config.get("cpu_offload_transformer_blocks", False)) and getattr(self.transformer, "training", False):
        _move_transformer_blocks_to_device(self.transformer, device)

    if not return_dict:
        return (image, lr_guidance_img_save) if model_config.get("joint_denoise", False) else (image, )

    # Ensure we're returning a consistent format
    if model_config.get("joint_denoise", False):
        return Flux2PipelineOutput(images=[image, lr_guidance_img_save])
    else:
        return Flux2PipelineOutput(images=image)