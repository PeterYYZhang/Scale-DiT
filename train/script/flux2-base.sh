# Specify the WAMDB API key
export WANDB_API_KEY='' # TODO: input your WANDB API key

# Specify the config file path
export XFL_CONFIG="./train/config/infer_flux-base.yaml"

echo "Starting run... using config: "
echo $XFL_CONFIG

export TOKENIZERS_PARALLELISM=true

accelerate launch --num_processes 1 --main_process_port $(( ( RANDOM % 10000 )  + 50000 )) -m src.train.train2 --disable_wandb