import os
import wandb
import torch
import lightning as L
from ..flux.generate2 import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.total_steps == 0:
            self.generate_a_sample(
                    trainer,
                    pl_module,
                    f"{self.save_path}/{self.run_name}/output",
                    f"lora_{self.total_steps}",
                    height=trainer.training_config["dataset"]["image_height"],
                    width=trainer.training_config["dataset"]["image_width"],
                    num_inference_steps=trainer.training_config.get("generation", {}).get("num_inference_steps", 14),
                    hr_inference_steps=trainer.training_config.get("generation", {}).get("hr_inference_steps", 14),
                    hr_guidance_scale=trainer.training_config.get("generation", {}).get("hr_guidance_scale", 4.5),
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                height=trainer.training_config["dataset"]["image_height"],
                width=trainer.training_config["dataset"]["image_width"],
                num_inference_steps=trainer.training_config.get("generation", {}).get("num_inference_steps", 14),
                hr_inference_steps=trainer.training_config.get("generation", {}).get("hr_inference_steps", 14),
                hr_guidance_scale=trainer.training_config.get("generation", {}).get("hr_guidance_scale", 4.5),
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        height,
        width,
        num_inference_steps=14,
        hr_inference_steps=14,
        hr_guidance_scale=4.5,
    ):

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        # default test prompts
        test_list = [ "Fashionista in avant-garde makeup with geometric face paint, high fashion editorial lighting, artistic beauty experimentation, contemporary style pushing boundaries",
                     "surreal dreamscape of floating islands connected by rainbow bridges, impossible waterfalls flowing upward into the sky, luminescent trees with crystal leaves, oversized butterflies with translucent wings, soft pastel clouds drifting between the islands, gravity-defying architecture with spiral towers, dreamy atmospheric lighting with lens flares, fantastical creatures roaming through the landscape, ethereal mist and magical particles in the air.",
                     "elegant Victorian living room with ornate furniture, Persian rugs, grandfather clock, fireplace with crackling flames, leather armchairs, bookshelves filled with classic literature, oil paintings in gilded frames, crystal chandelier, warm golden lighting, antique decor, sophisticated atmosphere.",
                     ]
        # Read test prompts from file
        if pl_module.model_config.get("train", False):     
            testlist_path = "/scratch/yuyao/Scale-DiT/prompt_new.txt" # you can put your own test prompts here (txt file)
            testlist_path = ""
        else:
            testlist_path = "/scratch/yuyao/Scale-DiT/prompt_new.txt"
            testlist_path = ""
        try:
            with open(testlist_path, 'r', encoding='utf-8') as f:
                test_list = [line.strip() for line in f if line.strip()]
            # test_list = test_list[750:]
        except FileNotFoundError:
            print(f"Warning: Test list file not found at {testlist_path}, using default prompts")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, prompt in enumerate(test_list):
            res = generate(
                pl_module.flux_pipe,
                prompt=prompt,
                height=height,
                width=width,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
                max_sequence_length=pl_module.model_config.get("text_seq_len", 512),
                save_path=save_path,
                num_inference_steps=num_inference_steps,
                hr_inference_steps=hr_inference_steps,
                hr_guidance_scale=hr_guidance_scale,
            )
            if pl_module.model_config.get("joint_denoise", False):
                # Handle all possible return formats
                images = res.images
                
                # Case 1: Direct tuple of images
                if isinstance(images, tuple) and len(images) == 2:
                    high_res_image, low_res_image = images
                # Case 2: List containing images
                elif isinstance(images, list):
                    if len(images) >= 2:
                        high_res_image = images[0]
                        low_res_image = images[1]
                    else:
                        # Fallback if we don't have two images
                        print("Warning: Expected two images but got fewer. Using the first image for both.")
                        high_res_image = images[0]
                        low_res_image = images[0]
                else:
                    # Unexpected format, use what we have
                    print(f"Warning: Unexpected image format: {type(images)}. Using as is.")
                    high_res_image = images
                    low_res_image = images
                    
                # Save high resolution image
                if isinstance(high_res_image, (list, tuple)):
                    # If it's still a container, take the first element
                    if len(high_res_image) > 0:
                        high_res_image = high_res_image[0]
                        
                # Save low resolution image
                if isinstance(low_res_image, (list, tuple)):
                    # If it's still a container, take the first element
                    if len(low_res_image) > 0:
                        low_res_image = low_res_image[0]
                        
                # Save the images
                high_res_image.save(
                    os.path.join(save_path, f"{file_name}_{prompt[:50]}_{i}.jpg")
                )
                low_res_image.save(
                    os.path.join(save_path, f"{file_name}_{prompt[:50]}_{i}_low_res_after.jpg")
                )
            else:
                res.images[0].save(
                    os.path.join(save_path, f"{file_name}_{prompt[:50]}_{i}.jpg")
                )