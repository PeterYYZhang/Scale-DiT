import os
import gc
import math
import torch
import psutil
import prodigyopt
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from ..flux.auto_encoder_kl import AutoencoderKL
from peft import LoraConfig, get_peft_model_state_dict
from ..flux.transformer2 import tranformer_forward
from ..flux.pipeline_tools import encode_images_tiled2, prepare_text_input, encode_images_tiled, prepare_text_input2, window_permutation


class FluxHierarchicalModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        max_memory_gb: float = 32.0,  # RAM limit for model loading
        cpu_offload: bool = False,    # Offload non-active components to CPU
        low_cpu_mem_usage: bool = True,  # Use memory-efficient loading
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.max_memory_gb = max_memory_gb
        self.cpu_offload = cpu_offload
        
        # Configure memory management
        self._setup_memory_management()
        
        # Load the Flux pipeline with memory optimization
        print(f"Loading Flux pipeline with max RAM: {max_memory_gb}GB, CPU offload: {cpu_offload}")
        self.flux_pipe: Flux2KleinPipeline = self._load_flux_pipeline_memory_efficient(
            flux_pipe_id, device, dtype, low_cpu_mem_usage
        )
        self.text_encoder = self.flux_pipe.text_encoder
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # self.flux_pipe.scheduler.config.use_dynamic_shifting = False
        # self.flux_pipe.scheduler.config.time_shift = 10
        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.flux_pipe.scheduler.config.max_image_seq_len = 65536*4
        
        self.to(device=device, dtype=dtype)

    def _materialize_meta_module_(self, module: nn.Module, device: str = "cpu") -> int:
        """
        Materialize any meta tensors created by low_cpu_mem_usage loading.

        Some checkpoints miss keys (newly initialized layers). Under low_cpu_mem_usage,
        those parameters can remain on the meta device, which makes subsequent `.to()`
        fail with "Cannot copy out of meta tensor".
        """
        materialized = 0
        for sub in module.modules():
            # Parameters
            for name, p in list(sub._parameters.items()):
                if p is None or not getattr(p, "is_meta", False):
                    continue

                new_t = torch.empty(p.shape, device=device, dtype=p.dtype)
                # Best-effort init consistent with common PyTorch modules.
                if name.endswith("bias"):
                    nn.init.zeros_(new_t)
                elif name.endswith("weight"):
                    if new_t.dim() == 1:
                        nn.init.ones_(new_t)
                    else:
                        nn.init.kaiming_uniform_(new_t, a=math.sqrt(5))
                else:
                    nn.init.normal_(new_t, mean=0.0, std=0.02)

                sub._parameters[name] = nn.Parameter(new_t, requires_grad=p.requires_grad)
                materialized += 1

            # Buffers
            for name, b in list(sub._buffers.items()):
                if b is None or not getattr(b, "is_meta", False):
                    continue
                sub._buffers[name] = torch.zeros(b.shape, device=device, dtype=b.dtype)
                materialized += 1

        return materialized

    def _materialize_meta_pipeline_(self, flux_pipe: Flux2KleinPipeline, device: str = "cpu") -> int:
        materialized = 0
        for attr in ("transformer", "text_encoder", "text_encoder_2", "vae", "vae2"):
            mod = getattr(flux_pipe, attr, None)
            if isinstance(mod, nn.Module):
                materialized += self._materialize_meta_module_(mod, device=device)
        return materialized

    def _setup_memory_management(self):
        """Configure PyTorch memory management settings"""
        # Set memory fraction for CUDA
        if torch.cuda.is_available():
            # Reserve some GPU memory for other processes
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            torch.cuda.empty_cache()
        
        # Configure memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"Memory management configured:")
        print(f"  - Max RAM limit: {self.max_memory_gb}GB")
        print(f"  - CPU offload: {self.cpu_offload}")
        print(f"  - Current RAM usage: {psutil.virtual_memory().percent:.1f}%")
        if torch.cuda.is_available():
            print(f"  - GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.1f}GB")
    
    def _check_memory_usage(self, stage: str = ""):
        """Monitor memory usage during loading"""
        ram_gb = psutil.virtual_memory().used / (1024**3)
        ram_percent = psutil.virtual_memory().percent
        
        print(f"Memory usage {stage}: RAM {ram_gb:.1f}GB ({ram_percent:.1f}%)")
        
        if ram_gb > self.max_memory_gb:
            print(f"WARNING: RAM usage ({ram_gb:.1f}GB) exceeds limit ({self.max_memory_gb}GB)")
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _load_flux_pipeline_memory_efficient(self, flux_pipe_id: str, device: str, dtype: torch.dtype, low_cpu_mem_usage: bool):
        """Load Flux pipeline with memory optimization"""
        self._check_memory_usage("before loading")
        
        try:
            # Load with memory-efficient settings
            loading_kwargs = {
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "torch_dtype": dtype,
            }
            
            # Add device map for memory management
            if self.cpu_offload:
                # Offload parts to CPU to save GPU memory
                loading_kwargs["device_map"] = "auto"
                loading_kwargs["max_memory"] = {0: "20GB", "cpu": f"{self.max_memory_gb}GB"}
            
            print(f"Loading pipeline with kwargs: {loading_kwargs}")
            # NOTE: `torch_dtype` is deprecated in diffusers; use `dtype`.
            if "torch_dtype" in loading_kwargs:
                loading_kwargs["dtype"] = loading_kwargs.pop("torch_dtype")
            flux_pipe = Flux2KleinPipeline.from_pretrained(flux_pipe_id, **loading_kwargs)

            # If low_cpu_mem_usage is enabled and the checkpoint misses some keys,
            # newly initialized params can remain on meta. Materialize them before any `.to()`.
            materialized = self._materialize_meta_pipeline_(flux_pipe, device="cpu")
            if materialized > 0:
                print(f"Materialized {materialized} meta tensors after loading.")
            
            self._check_memory_usage("after pipeline loading")
            
            # Move to device if not using device_map
            if not self.cpu_offload:
                flux_pipe = flux_pipe.to(device)
                self._check_memory_usage("after moving to device")
            
            return flux_pipe
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            print("Trying fallback loading method...")
            
            # Fallback to sequential loading
            return self._load_flux_pipeline_fallback(flux_pipe_id, device, dtype)
    
    def _load_flux_pipeline_fallback(self, flux_pipe_id: str, device: str, dtype: torch.dtype):
        """Fallback loading method with aggressive memory management"""
        print("Using fallback loading method with aggressive memory management")
        
        # Clear cache before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load with minimal memory footprint
        flux_pipe = Flux2KleinPipeline.from_pretrained(
            flux_pipe_id,
            dtype=dtype,
            low_cpu_mem_usage=True, # Load to CPU first
        )

        materialized = self._materialize_meta_pipeline_(flux_pipe, device="cpu")
        if materialized > 0:
            print(f"Materialized {materialized} meta tensors after fallback loading.")
        
        self._check_memory_usage("after CPU loading")
        
        # Move components to GPU one by one if needed
        if device != "cpu" and not self.cpu_offload:
            print("Moving components to GPU sequentially...")
            
            # Move transformer (main component)
            flux_pipe.transformer = flux_pipe.transformer.to(device)
            gc.collect()
            torch.cuda.empty_cache()
            self._check_memory_usage("after transformer to GPU")
            
            # Keep text encoders and VAE on CPU if memory is tight
            if psutil.virtual_memory().percent > 80:
                print("Keeping text encoders and VAE on CPU due to memory constraints")
            else:
                flux_pipe.text_encoder = flux_pipe.text_encoder.to(device)
                if hasattr(flux_pipe, "text_encoder_2") and flux_pipe.text_encoder_2 is not None:
                    flux_pipe.text_encoder_2 = flux_pipe.text_encoder_2.to(device)
                flux_pipe.vae = flux_pipe.vae.to(device)
                self._check_memory_usage("after all components to GPU")
        
        return flux_pipe

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # Determine if lora_path is a file or directory
            if os.path.isfile(lora_path):
                # If it's a file (like pytorch_lora_weights.safetensors), get the directory
                lora_dir = os.path.dirname(lora_path)
                lora_file = lora_path
            else:
                # If it's a directory, use it directly
                lora_dir = lora_path
                lora_file = lora_path
            
            # load the lora weights from the path
            self.flux_pipe.load_lora_weights(
                lora_file,
                adapter_name="default",
            )
            print(f"Loaded LoRA weights from {lora_file}")
            
            # Find LoRA parameters by name (they typically contain 'lora' in their name)
            lora_layers = []
            for name, param in self.transformer.named_parameters():
                if 'lora' in name.lower():
                    lora_layers.append(param)
            print(f"Number of LoRA parameters found: {len(lora_layers)}")
            print(f"Number of trainable parameters: {sum(p.numel() for p in lora_layers)}")
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        # Ensure path is a directory
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            
        Flux2KleinPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters (LoRA layers + cross attention weights)
        self.trainable_params = self.lora_layers
        
        # Debug information
        print(f"Number of LoRA layers stored: {len(self.lora_layers)}")
        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)
        print(f"Total trainable parameters: {sum(p.numel() for p in self.trainable_params if p.requires_grad)}")
        
        if len(self.trainable_params) == 0:
            print("ERROR: No trainable parameters found!")
            print("All transformer parameters:")
            for name, param in self.transformer.named_parameters():
                print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            raise ValueError("No trainable parameters found for optimizer")

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        # Memory optimization: clear cache before each step
        if batch_idx % 10 == 0:  # Every 10 steps
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Use automatic mixed precision if configured
        if self.model_config.get("use_amp", False):
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                step_loss = self.step(batch)
        else:
            step_loss = self.step(batch)
        
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        
        # Memory monitoring
        if batch_idx % 50 == 0:  # Every 50 steps
            ram_percent = psutil.virtual_memory().percent
            if ram_percent > 85:
                print(f"High RAM usage detected: {ram_percent:.1f}%")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        prompts = batch["description"]
        img_lr = imgs.clone()
        # Resize low resolution image to joint_denoise_size
        joint_denoise_size = self.model_config.get("joint_denoise_size", 256)
        img_lr = torch.nn.functional.interpolate(
            img_lr, size=(joint_denoise_size[0], joint_denoise_size[1]), mode="bicubic", align_corners=False
        )
        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images_tiled2(self.flux_pipe, imgs)

            if self.model_config.get("joint_denoise", False):
                x_0_lowres, img_ids_lowres = encode_images_tiled2(self.flux_pipe, img_lr)

            if self.model_config.get("permute_window_first", False):
                hr_permutation_idx = window_permutation(
                    H=self.model_config.get("image_size", [1024, 1024])[0]//16,
                    W=self.model_config.get("image_size", [1024, 1024])[1]//16,
                    Wh=16,
                    Ww=16,
                )
                x_0 = x_0[:, hr_permutation_idx, ]
                img_ids = img_ids[:, hr_permutation_idx,]
            if self.model_config.get("joint_denoise_size", [256, 256])[0] > 256:
                lr_permutation_idx = window_permutation(
                    H=self.model_config.get("joint_denoise_size", [256, 256])[0]//16,
                    W=self.model_config.get("joint_denoise_size", [256, 256])[1]//16,
                    Wh=4,
                    Ww=4,
                )
                x_0_lowres = x_0_lowres[:, lr_permutation_idx, ]
                img_ids_lowres = img_ids_lowres[:, lr_permutation_idx,]
            
            # Prepare text input
            prompt_embeds, text_ids = self.flux_pipe.encode_prompt(
                prompt=prompts,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("text_seq_len", 512),
                text_encoder_out_layers=self.model_config.get("text_encoder_out_layers", (9, 18, 27)),
            ) 
              
            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype) 

            
            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )


        # Remove batch dim when 3D; transformer expects 2D
        lr_guidance = x_0_lowres[0] if x_0_lowres.ndim == 3 else x_0_lowres
        lr_ids = img_ids_lowres[0] if img_ids_lowres.ndim == 3 else img_ids_lowres
        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            # Newly added codes
            low_res_guidance=lr_guidance,
            low_res_img_ids=lr_ids,
            low_res_timestep=torch.zeros_like(t),
        )
        pred = transformer_out[0]
        target = x_1 - x_0
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
        if self.model_config.get("consistency_loss", False):
            pass
        self.last_t = t.mean().item()
        return loss