import torch
import numpy as np
from typing import Optional, Dict, Any
from .lora_controller import enable_lora
from .pipeline_tools import FluxPosEmbedForward, Flux2PosEmbedForward
from .block2 import block_forward, single_block_forward
from .block2_train import block_forward as block_forward_train, single_block_forward as single_block_forward_train
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel,
    Transformer2DModelOutput,
    Flux2PosEmbed,
)
from diffusers.utils import (scale_lora_layers, unscale_lora_layers, logger, USE_PEFT_BACKEND)


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
    transformer: Flux2Transformer2DModel,
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

    if low_res_guidance is not None:
        if low_res_guidance.ndim == 3:
            low_res_guidance = low_res_guidance[0]
        low_res_guidance = self.x_embedder(low_res_guidance)

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
    else:
        guidance = None
        temb = self.time_guidance_embed(timestep, None)
        
    if low_res_timestep is not None:
        # `low_res_timestep` can be a python scalar or a tensor; use `as_tensor` to avoid copy-construct warnings.
        low_res_timestep_tensor = torch.as_tensor(
            low_res_timestep, device=timestep.device, dtype=hidden_states.dtype
        ) * torch.ones_like(timestep, dtype=hidden_states.dtype)
        if guidance is not None:
            low_res_temb = self.time_guidance_embed(low_res_timestep_tensor, guidance)
        else:
            low_res_temb = self.time_guidance_embed(low_res_timestep_tensor, None)
    else:
        low_res_temb = None

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)
    if low_res_guidance is not None:
        double_stream_mod_lr = self.double_stream_modulation_img(low_res_temb)
        single_stream_mod_lr = self.single_stream_modulation(low_res_temb)
    else:
        double_stream_mod_lr = None
        single_stream_mod_lr = None

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    # import pdb; pdb.set_trace()
    if txt_ids.ndim == 3:
        # logger.warning(
        #     "Passing `txt_ids` 3d torch.Tensor is deprecated."
        #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
        # )
        txt_ids = txt_ids[0]

    if img_ids.ndim == 3:
        # logger.warning(
        #     "Passing `img_ids` 3d torch.Tensor is deprecated."
        #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
        # )
        img_ids = img_ids[0]
    if low_res_img_ids is not None:
        if low_res_img_ids.ndim == 3:
            low_res_img_ids = low_res_img_ids[0]
        if model_config.get("scale_PE", False):
            hr_lr_scale = model_config.get("image_size", [1024, 1024])[0] // model_config.get("joint_denoise_size", [256, 256])[0]
            low_res_img_ids = low_res_img_ids*hr_lr_scale
        
    # Build a single RoPE embedding over concatenated token ids (txt + img [+ optional LR img]).
    # IMPORTANT: always pass `theta` explicitly. The official FLUX.2 uses `pos_embed.theta` (rope_theta, default 2000),
    # but our helper defaults to 10000 if omitted, which can destabilize inference and produce noisy images.
    ids = (
        torch.cat((txt_ids, img_ids, low_res_img_ids), dim=0)
        if low_res_img_ids is not None
        else torch.cat((txt_ids, img_ids), dim=0)
    )
    if low_res_img_ids is not None:
        concat_rotary_emb = Flux2PosEmbedForward(self.pos_embed, ids, 
        model_config.get("ntk_factor", 1.0), )
    else:
        concat_rotary_emb = self.pos_embed(ids)       


    
    # Double stream 
    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            use_sage = model_config.get("use_sage_blocksparse", False)

            # IMPORTANT: bind loop vars as defaults. Checkpoint will re-call this closure during backward,
            # and Python would otherwise late-bind `block`/`index_block` to the *last* loop values.
            def _block_forward_ckpt(h, enc_h, lr_g, block=block, index_block=index_block):
                # Safety: some pipelines/configs may place single-stream blocks in `transformer_blocks`.
                # Double-stream blocks have `norm1_context`; single-stream blocks only have `norm`.
                if hasattr(block, "norm1_context"):
                    forward_fn = block_forward if use_sage else block_forward_train
                    return forward_fn(
                        block,
                        model_config=model_config,
                        hidden_states=h,
                        encoder_hidden_states=enc_h,
                        image_rotary_emb=concat_rotary_emb,
                        low_res_guidance=lr_g,
                        block_idx=index_block,
                        timestep=timestep,
                        double_stream_mod_img=double_stream_mod_img,
                        double_stream_mod_txt=double_stream_mod_txt,
                        double_stream_mod_lr=double_stream_mod_lr,
                    )

                forward_fn = single_block_forward if use_sage else single_block_forward_train
                text_len = enc_h.shape[1]
                h_cat = torch.cat([enc_h, h], dim=1)
                h_cat, lr_g = forward_fn(
                    block,
                    model_config=model_config,
                    hidden_states=h_cat,
                    temb=single_stream_mod,
                    low_res_temb=single_stream_mod_lr,
                    image_rotary_emb=concat_rotary_emb,
                    low_res_guidance=lr_g,
                    block_idx=index_block,
                    timestep=timestep,
                )
                enc_h, h = h_cat[:, :text_len], h_cat[:, text_len:]
                return enc_h, h, lr_g

            checkpoint_output = torch.utils.checkpoint.checkpoint(
                _block_forward_ckpt,
                hidden_states,
                encoder_hidden_states,
                low_res_guidance,
                use_reentrant=False,
            )
            if low_res_guidance is None:
                encoder_hidden_states, hidden_states, _ = checkpoint_output
            else:
                encoder_hidden_states, hidden_states, low_res_guidance = checkpoint_output
        else:
            use_sage = model_config.get("use_sage_blocksparse", False)
            if hasattr(block, "norm1_context"):
                forward_fn = block_forward if use_sage else block_forward_train
                checkpoint_output = forward_fn(
                    block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=concat_rotary_emb,
                    low_res_guidance=low_res_guidance,
                    block_idx=index_block,
                    timestep=timestep,
                    double_stream_mod_img=double_stream_mod_img,
                    double_stream_mod_txt=double_stream_mod_txt,
                    double_stream_mod_lr=double_stream_mod_lr,
                )
                if low_res_guidance is None:
                    encoder_hidden_states, hidden_states, _ = checkpoint_output
                else:
                    encoder_hidden_states, hidden_states, low_res_guidance = checkpoint_output
            else:
                forward_fn = single_block_forward if use_sage else single_block_forward_train
                text_len = encoder_hidden_states.shape[1]
                h_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                h_cat, low_res_guidance = forward_fn(
                    block,
                    model_config=model_config,
                    hidden_states=h_cat,
                    temb=single_stream_mod,
                    low_res_temb=single_stream_mod_lr,
                    image_rotary_emb=concat_rotary_emb,
                    low_res_guidance=low_res_guidance,
                    block_idx=index_block,
                    timestep=timestep,
                )
                encoder_hidden_states, hidden_states = h_cat[:, :text_len], h_cat[:, text_len:]

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            use_sage = model_config.get("use_sage_blocksparse", False)

            # Same late-binding issue as above for backward recomputation.
            def _single_block_forward_ckpt(h, lr_g, block=block, index_block=index_block):
                forward_fn = single_block_forward if use_sage else single_block_forward_train
                return forward_fn(
                    block,
                    model_config=model_config,
                    hidden_states=h,
                    temb=single_stream_mod,
                    low_res_temb=single_stream_mod_lr,
                    image_rotary_emb=concat_rotary_emb,
                    low_res_guidance=lr_g,
                    block_idx=index_block,
                    timestep=timestep,
                )

            checkpoint_output = torch.utils.checkpoint.checkpoint(
                _single_block_forward_ckpt,
                hidden_states,
                low_res_guidance,
                use_reentrant=False,
            )
            if low_res_guidance is None:
                hidden_states, _ = checkpoint_output
            else:
                hidden_states, low_res_guidance = checkpoint_output
        else:
            if model_config.get("use_sage_blocksparse", False):
                checkpoint_output = single_block_forward(
                    block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    temb=single_stream_mod,
                    low_res_temb=single_stream_mod_lr,
                    image_rotary_emb=concat_rotary_emb,
                    low_res_guidance=low_res_guidance,
                    block_idx=index_block,
                    timestep=timestep,
                )
            else:
                checkpoint_output = single_block_forward_train(
                    block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    temb=single_stream_mod,
                    low_res_temb=single_stream_mod_lr,
                    image_rotary_emb=concat_rotary_emb,
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