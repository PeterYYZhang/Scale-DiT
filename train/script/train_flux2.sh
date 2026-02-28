# Specify the WAMDB API key
export WANDB_API_KEY='525ee645de7b6dc49d6588443446f728fbfb90b1' # TODO: input your WANDB API key

# Specify the config file path
export XFL_CONFIG="./train/config/train_flux2.yaml"

echo "Starting run... using config: "
echo $XFL_CONFIG

export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port $(( ( RANDOM % 10000 )  + 50000 )) -m src.train.train2 #--disable_wandb