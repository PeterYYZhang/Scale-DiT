import os 
import time
import yaml
import torch
import lightning as L
import torchvision.transforms as T
from .data import ImageConditionDataset
from torch.utils.data import DataLoader
from .callbacks import TrainingCallback
from .model import FluxHierarchicalModel
from datasets import load_dataset, load_from_disk
from lightning.pytorch.strategies import DDPStrategy


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def init_wandb(wandb_config, run_name):
    import wandb
    
    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = training_config.get("run_name", "try") + time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataset and dataloader
    dataset_config = training_config["dataset"]
    filtered_cache_dir = "data/filtered"
    # Check if filtered dataset already exists
    if os.path.exists(filtered_cache_dir):
        print(f"Loading filtered dataset from {filtered_cache_dir}")
        dataset = load_from_disk(filtered_cache_dir)
        print(f"Filtered dataset size: {len(dataset)}")
    else:
        print("Loading original dataset and filtering...")
        dataset = load_dataset(
            "webdataset",
            data_files=dataset_config.get("urls", []),
            split="train",
            cache_dir="datasets/extracted_data",
            num_proc=32,
        )

        # Filter dataset: TODO: add filtering logic here
        def filter_none_values(example)->bool:
            pass

        # print(f"Original dataset size: {len(dataset)}")
        # dataset = dataset.filter(filter_none_values)
        # print(f"Filtered dataset size: {len(dataset)}")
        
        # Save filtered dataset to cache
        os.makedirs(os.path.dirname(filtered_cache_dir), exist_ok=True)
        print(f"Saving filtered dataset to {filtered_cache_dir}")
        dataset.save_to_disk(filtered_cache_dir)

    dataset = ImageConditionDataset(
        base_dataset=dataset,
        target_size=dataset_config.get("target_size", 1024),
        drop_text_prob=dataset_config.get("drop_text_prob", 0.1),
    )

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = FluxHierarchicalModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        lora_path=config["lora_path"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
        # The 'logger' argument in PyTorch Lightning's Trainer controls experiment logging.
        # Setting logger=False disables all logging (e.g., to TensorBoard, WandB, CSV, etc.).
        # If you want to enable logging, you can pass a logger instance (e.g., WandbLogger).
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    def parse_disable_wandb_flag():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
        args, unknown = parser.parse_known_args()
        return args.disable_wandb

    if __name__ == "__main__":
        disable_wandb = parse_disable_wandb_flag()
        if disable_wandb:
            os.environ["WANDB_MODE"] = "disabled"
        main()