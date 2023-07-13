export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
device_id=7

num_samples=10
num_classes=1000
vae_ckpt_path=output/finetune_pokemon
vae_config_path=vae_config_path
model_path=/model_path

export RANK_SIZE=1;export DEVICE_ID=$device_id; \
python sample_image.py \
    --model_path=$model_path \
    --num_samples=$num_samples \
    --num_classes=$num_classes \
    --vae_ckpt_path=$vae_ckpt_path \
    --vae_config_path=$vae_config_path
