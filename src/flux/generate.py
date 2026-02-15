import torch
import yaml, os
import time
import csv
import torch.nn.functional as F
import pywt
from PIL import Image
from diffusers.pipelines import FluxPipeline
from .transformer import tranformer_forward
from typing import List, Union, Optional, Dict, Any, Callable
from .pipeline_tools import (
    save_image,
    window_permutation,
    inverse_permutation,
    encode_images_tiled,
)

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)

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
    pipeline: FluxPipeline,
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

    self.vae.enable_tiling()
    guidance_scale = model_config.get("lr_guidance_scale", guidance_scale)
    
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    original_image_size = model_config.get("image_size", (height, width))
    model_config["image_size"] = (height, width)

    os.makedirs(save_path, exist_ok=True)
    
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        model_config.get("joint_denoise_size", [256, 256])[0],
        model_config.get("joint_denoise_size", [256, 256])[1],
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Ensure text encoder is on the correct device before encoding
    if hasattr(self, 'text_encoder') and self.text_encoder is not None:
        self.text_encoder.to(device)
    if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
        self.text_encoder_2.to(device)

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # Offload the encoder to CPU after encoding
    if hasattr(self, 'text_encoder') and self.text_encoder is not None:
        self.text_encoder.to("cpu")
    if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
        self.text_encoder_2.to("cpu")

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        model_config.get("joint_denoise_size", [256, 256])[0],
        model_config.get("joint_denoise_size", [256, 256])[1],
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    low_res_latents_ids = latent_image_ids
    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None 
    # 6. Denoising loop for low-res image
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = tranformer_forward(
                self.transformer,
                model_config=model_config,
                hidden_states=latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

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
    # low_res_path = "/scratch/yuyao/Scale-DiT/flux_linear_3.png"
    low_res_path = None
    if low_res_path is not None:
        joint_size = model_config.get("joint_denoise_size", [256, 256])
        image = Image.open(low_res_path).convert("RGB")

        # Pad image to 512x512 instead of center crop
        target_size = 512
        width, height = image.size
        target_w, target_h = height, height
        left = (width - target_w) / 2
        top = (height - target_h) / 2
        right = (width + target_w) / 2
        bottom = (height + target_h) / 2
        image = image.crop((left, top, right, bottom)).resize((joint_size[0], joint_size[1]), resample=Image.BICUBIC)

        latents, _ = encode_images_tiled(self, image)

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
    )
    
    # Configure model to use BlockSparseAttention for high-res generation
    if model_config.get("use_block_sparse_attention", False):
        model_config["use_flash_attention"] = False 
        model_config["use_block_sparse_attention"] = True
    # 7. Denoising loop for high-res image
    # Build HR guidance latents efficiently (avoid VAE decode -> resize -> VAE encode).
    # We upsample *in latent space* then repack to token form.
    height, width = model_config.get("image_size", (height, width))[0], model_config.get("image_size", (height, width))[1]
    joint_h = model_config.get("joint_denoise_size", [256, 256])[0]
    joint_w = model_config.get("joint_denoise_size", [256, 256])[1]

    lr_latents_spatial = self._unpack_latents(latents, joint_h, joint_w, self.vae_scale_factor)
    # Target latent spatial size used by FLUX packing.
    pack_h = (height // self.vae_scale_factor) * 2
    pack_w = (width // self.vae_scale_factor) * 2
    hr_latents_spatial = F.interpolate(lr_latents_spatial, size=(pack_h, pack_w), mode="bicubic", align_corners=False)
    latents_guidance = self._pack_latents(
        hr_latents_spatial,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        pack_h,
        pack_w,
    ).to(device=device)
    # For HR generation, we want `latents` / `latent_image_ids` to reflect the HR token layout.
    # If `interpolation_init` is enabled, we will start from these guidance latents; otherwise
    # `prepare_latents` below will overwrite `latents` with random noise latents.
    latents = latents_guidance

    # Prepare HR latent ids (same logic as `encode_images_tiled`, but without re-encoding).
    latent_image_ids = self._prepare_latent_image_ids(
        hr_latents_spatial.shape[0],
        hr_latents_spatial.shape[2],
        hr_latents_spatial.shape[3],
        device,
        self.dtype,
    )
    if latents_guidance.shape[1] != latent_image_ids.shape[0]:
        latent_image_ids = self._prepare_latent_image_ids(
            hr_latents_spatial.shape[0],
            hr_latents_spatial.shape[2] // 2,
            hr_latents_spatial.shape[3] // 2,
            device,
            self.dtype,
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
        H=model_config.get("image_size", [1024, 1024])[0]//16,
        W=model_config.get("image_size", [1024, 1024])[1]//16,
        Wh=16,
        Ww=16,
    )
    inv_hr_permutation_idx = inverse_permutation(hr_permutation_idx)
    latent_image_ids = latent_image_ids[hr_permutation_idx]
    latents = latents[:, hr_permutation_idx, ]
    # Align guidance tokens to the same permutation order used for HR denoising.
    latents_guidance = latents_guidance[:, hr_permutation_idx, ].to(dtype=latents.dtype, device=device)
    if model_config.get("joint_denoise_size", [256, 256])[0] > 256:
        lr_permutation_idx = window_permutation(
            H=model_config.get("joint_denoise_size", [256, 256])[0]//16,
            W=model_config.get("joint_denoise_size", [256, 256])[1]//16,
            Wh=4,
            Ww=4,
        )
        lr_latents_copy = lr_latents_copy[:, lr_permutation_idx, ]
        low_res_latents_ids = low_res_latents_ids[lr_permutation_idx]
    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.tensor([hr_guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None 

    # Projected-flow LR guidance configuration (cosine_shift).
    guidance_schedule = model_config.get("guidance_schedule", "cosine_shift")
    dwt_level = int(model_config.get("dwt_level", 1))
    dwt_wavelet = model_config.get("dwt_wavelet", "haar")
    pack_h = (height // self.vae_scale_factor) * 2
    pack_w = (width // self.vae_scale_factor) * 2

    def _lowpass_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D) in *permuted* HR order.
        Returns: low-frequency tokens (B, N, D) in the same permuted order, computed via DWT.
        """
        tokens_unperm = tokens[:, inv_hr_permutation_idx, :]
        spatial = self._unpack_latents(tokens_unperm, height, width, self.vae_scale_factor)
        spatial_low = split_frequency_components_dwt(spatial, wavelet=dwt_wavelet, level=dwt_level)
        tokens_low_unperm = self._pack_latents(
            spatial_low,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            pack_h,
            pack_w,
        )
        return tokens_low_unperm[:, hr_permutation_idx, :]

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

            noise_pred = tranformer_forward(
                self.transformer,
                model_config=model_config,
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
                # Newly added codes
                low_res_guidance=lr_latents_copy,
                low_res_img_ids=low_res_latents_ids,
                low_res_timestep=0,
            )[0]
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
                        * (torch.tensor(float(i+hr_inference_steps), device=device, dtype=latents.dtype) / torch.tensor(float(num_inference_steps), device=device, dtype=latents.dtype))
                    )
                )

                if guidance_schedule == "cosine_decay":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + alpha * (fp_v_low - noise_pred_low)
                elif guidance_schedule == "cosine_hp_decay":
                    noise_pred = noise_pred + alpha * (fp_v - noise_pred)
                elif guidance_schedule == "cosine_shift":
                    fp_v_low = _lowpass_tokens(fp_v)
                    noise_pred_low = _lowpass_tokens(noise_pred)
                    noise_pred = noise_pred + alpha * (fp_v - noise_pred) + (1.0 - alpha) * (fp_v_low - noise_pred_low)
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
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        if model_config.get("finetuned_vae", False):
            image = self.vae2.decode(latents.to(self.vae2.dtype), return_dict=False)[0]
        else:
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

    if not return_dict:
        return (image, lr_guidance_img_save) if model_config.get("joint_denoise", False) else (image, )

    # Ensure we're returning a consistent format
    if model_config.get("joint_denoise", False):
        return FluxPipelineOutput(images=[image, lr_guidance_img_save])
    else:
        return FluxPipelineOutput(images=image)