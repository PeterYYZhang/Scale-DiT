# Specify the WAMDB API key
export WANDB_API_KEY='' # TODO: input your WANDB API key

# Specify the config file path
export XFL_CONFIG="./train/config/inference.yaml"

echo "Starting run... using config: "
echo $XFL_CONFIG

export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port $(( ( RANDOM % 10000 )  + 50000 )) -m src.train.train --disable_wandb