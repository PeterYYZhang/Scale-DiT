import torch
import numpy as np
from typing import Optional, Dict, Any
from .lora_controller import enable_lora
from .pipeline_tools import FluxPosEmbedForward
from .block import block_forward, single_block_forward
from .block_train import block_forward as block_forward_train, single_block_forward as single_block_forward_train
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)



def prepare_params(
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
    # Newly added codes
    low_res_guidance: torch.Tensor = None,
    low_res_img_ids: torch.Tensor = None,
    low_res_timestep: torch.Tensor = None,
    **kwargs: dict,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
        # Newly added codes 
        low_res_guidance,
        low_res_img_ids,
        low_res_timestep,
    )

def tranformer_forward(
    transformer: FluxTransformer2DModel,
    model_config: Optional[Dict[str, Any]] = {},
    **params: dict,
):
    self = transformer

    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
        # Newly added codes
        low_res_guidance,
        low_res_img_ids,
        low_res_timestep,
    ) = prepare_params(**params)
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    with enable_lora((self.x_embedder,), False):
    # with enable_only_lora((self.x_embedder,), "urae"):
        hidden_states = self.x_embedder(hidden_states)
    if low_res_guidance is not None:     # need to use lora here
        low_res_guidance = self.x_embedder(low_res_guidance)

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
    else:
        guidance = None
        temb = self.time_text_embed(timestep, pooled_projection=pooled_projections)
    if low_res_timestep is not None:
        low_res_timestep_tensor = low_res_timestep * torch.ones_like(timestep)
        if guidance is not None:
            low_res_temb = self.time_text_embed(low_res_timestep_tensor, guidance, pooled_projections)
        else:
            low_res_temb = self.time_text_embed(low_res_timestep_tensor, pooled_projection=pooled_projections)
    else:
        low_res_temb = None

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
    if low_res_img_ids is not None:
        if low_res_img_ids.ndim == 3:
            logger.warning(
                "Passing `low_res_guidance` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            low_res_img_ids = low_res_img_ids[0]
        if model_config.get("scale_PE", False):
            hr_lr_scale = model_config.get("image_size", [1024, 1024])[0] // model_config.get("joint_denoise_size", [256, 256])[0]
            low_res_img_ids = low_res_img_ids*hr_lr_scale
        
    ids = torch.cat((txt_ids, img_ids, low_res_img_ids), dim=0) if low_res_img_ids is not None else torch.cat((txt_ids, img_ids), dim=0)
    if low_res_img_ids is not None:
        image_rotary_emb = FluxPosEmbedForward(self.pos_embed, ids, model_config.get("ntk_factor", 1.0))
    else:
        image_rotary_emb = self.pos_embed(ids)



    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            checkpoint_output = torch.utils.checkpoint.checkpoint(
                block_forward_train if not model_config.get("use_sage_blocksparse", False) else block_forward,
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                low_res_temb=low_res_temb,
                image_rotary_emb=image_rotary_emb,
                # Newly added codes
                low_res_guidance=low_res_guidance,
                block_idx=index_block,
                timestep=timestep,
                **ckpt_kwargs,
            )
            if low_res_guidance is None:
                encoder_hidden_states, hidden_states, _ = checkpoint_output
            else:
                encoder_hidden_states, hidden_states, low_res_guidance = checkpoint_output

        else:
            checkpoint_output = block_forward_train if not model_config.get("use_sage_blocksparse", False) else block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                low_res_temb=low_res_temb,
                image_rotary_emb=image_rotary_emb,
                # Newly added codes
                low_res_guidance=low_res_guidance,
                block_idx=index_block,
                timestep=timestep,
                )
            if low_res_guidance is None:
                encoder_hidden_states, hidden_states, _ = checkpoint_output
            else:
                encoder_hidden_states, hidden_states, low_res_guidance = checkpoint_output

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_block_samples[index_block // interval_control]
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            checkpoint_output = torch.utils.checkpoint.checkpoint(
                single_block_forward_train if not model_config.get("use_sage_blocksparse", False) else single_block_forward,
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                low_res_temb=low_res_temb,
                image_rotary_emb=image_rotary_emb,
                # Newly added codes
                low_res_guidance=low_res_guidance,
                block_idx=index_block,
                timestep=timestep,
                **ckpt_kwargs,
            )
            if low_res_guidance is None:
                hidden_states, _ = checkpoint_output
            else:
                hidden_states, low_res_guidance = checkpoint_output

        else:
            checkpoint_output = single_block_forward_train if not model_config.get("use_sage_blocksparse", False) else single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                low_res_temb=low_res_temb,
                image_rotary_emb=image_rotary_emb,
                # Newly added codes
                low_res_guidance=low_res_guidance,
                block_idx=index_block,
                timestep=timestep,
            )
            if low_res_guidance is None:
                hidden_states, _ = checkpoint_output
            else:
                hidden_states, low_res_guidance = checkpoint_output

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)