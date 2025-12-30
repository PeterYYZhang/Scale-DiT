import torch
import yaml, os
import torch.nn.functional as F
from diffusers.pipelines import FluxPipeline
from .transformer import tranformer_forward
from typing import List, Union, Optional, Dict, Any, Callable
from .pipeline_tools import (save_image,
                            encode_vae_latents,
                            window_permutation,
                            inverse_permutation,
                            encode_images_tiled,
                            gaussian_blur_image_sharpening)

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)


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


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})
    seed_everything(model_config.get("seed", 10086))
    self = pipeline
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

    # Save low-res image
    lr_latents_copy = latents.clone()
    save_image(self, latents, model_config.get("joint_denoise_size", [256, 256])[0], 
               model_config.get("joint_denoise_size", [256, 256])[1], 
               os.path.join(save_path, f"low_res_image_{prompt[:50]}.png"), output_type="pil")
    
    # Configure model to use BlockSparseAttention for high-res generation
    if model_config.get("use_block_sparse_attention", False):
        model_config["use_flash_attention"] = False 
        model_config["use_block_sparse_attention"] = True
    # 7. Denoising loop for high-res image
    # upsample the low-res image to the original size
    lr_guidance = self._unpack_latents(latents, 
            model_config.get("joint_denoise_size", [256, 256])[0], 
            model_config.get("joint_denoise_size", [256, 256])[1], self.vae_scale_factor)
    lr_guidance = (lr_guidance / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    lr_guidance_img = self.vae.decode(lr_guidance.to(self.vae.dtype), return_dict=False)[0]
    lr_guidance_img = F.interpolate(
        lr_guidance_img,
        size=original_image_size,
        mode="bicubic",
        align_corners=False,
    )
    # TODO: sharpen the interpolated LR image 
    lr_guidance_img_save = gaussian_blur_image_sharpening(lr_guidance_img)
    lr_guidance_img_save = self.image_processor.postprocess(lr_guidance_img_save, output_type="pil")

    height, width = model_config.get("image_size", (height, width))[0], model_config.get("image_size", (height, width))[1]
    latents, latent_image_ids = encode_images_tiled(self, lr_guidance_img_save)
    latent_image_ids = latent_image_ids.to(device)
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
    
    noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
    if model_config.get("interpolation_init", False):
        latents = self.scheduler.scale_noise(latents, dlfg_timesteps[None, 0], noise,).to(self.transformer.dtype)
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
    latent_image_ids = latent_image_ids[hr_permutation_idx]
    latents = latents[:, hr_permutation_idx, ]
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
            noise_pred = noise_pred
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

            # call the callback, if provided
            if i == len(dlfg_timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
    inv_hr_permutation_idx = inverse_permutation(hr_permutation_idx)
    latents = latents[:, inv_hr_permutation_idx, ]
    self.vae.enable_tiling()
    self.vae.enable_slicing()
    # 8. Save high-res image
    if output_type == "latent":
        image = latents
    else:
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