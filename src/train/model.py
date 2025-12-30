import os
import gc
import torch
import psutil
import prodigyopt
import lightning as L
import torch.nn.functional as F
from diffusers.pipelines import FluxPipeline
from ..flux.auto_encoder_kl import AutoencoderKL
from peft import LoraConfig, get_peft_model_state_dict
from ..flux.transformer import tranformer_forward
from ..flux.pipeline_tools import prepare_text_input, encode_images_tiled


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
        self.flux_pipe: FluxPipeline = self._load_flux_pipeline_memory_efficient(
            flux_pipe_id, device, dtype, low_cpu_mem_usage
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        if self.model_config.get("finetuned_vae", False):
            local_vae = AutoencoderKL.from_pretrained(self.model_config.get("finetuned_vae_path"), subfolder="vae", torch_dtype=torch.bfloat16)
            self.flux_pipe.vae2 = local_vae
            self.flux_pipe.vae2.to(device).to(dtype)
            # local_vae = AutoencoderKL.from_pretrained("Owen777/UltraFlux-v1",subfolder="vae", torch_dtype=torch.bfloat16)
        self.flux_pipe.scheduler.config.use_dynamic_shifting = False
        self.flux_pipe.scheduler.config.time_shift = 10
        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.flux_pipe.scheduler.config.max_image_seq_len = 65536
        
        self.to(device).to(dtype)

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
            flux_pipe = FluxPipeline.from_pretrained(flux_pipe_id, **loading_kwargs)
            
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
        flux_pipe = FluxPipeline.from_pretrained(
            flux_pipe_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Load to CPU first
        )
        
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
            
        FluxPipeline.save_lora_weights(
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
            x_0, img_ids = encode_images_tiled(self.flux_pipe, imgs)

            if self.model_config.get("joint_denoise", False):
                x_0_lowres, img_ids_lowres = encode_images_tiled(self.flux_pipe, img_lr)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts, max_sequence_length=self.model_config.get("text_seq_len", 512)
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

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            # Newly added codes
            low_res_guidance=x_0_lowres,
            low_res_img_ids=img_ids_lowres,
            low_res_timestep=torch.zeros_like(t),
        )
        pred = transformer_out[0]
        target = x_1 - x_0
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
        if self.model_config.get("consistency_loss", False):
            pass
        self.last_t = t.mean().item()
        return loss