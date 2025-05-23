#!/bin/bash

uid="$(date +%Y%m%d_%H%M%S)"
base_models=(
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
)
lrs=(1e-5)
min_lr=0
epochss=(1)
weight_decay=1e-4             # -> the same training pipe as slurm_training
micro_batch_size=1        # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=4
push_to_hub=False
mode="safetune"
dataset="UWNSL/SafeChain"

PORT_START=12340
PORT_END=12400
# Function to find a free port
find_free_port() {
    for port in $(seq $PORT_START $PORT_END); do
        if ! lsof -i:$port > /dev/null; then
            echo $port
            return 0
        fi
    done
    echo "No free ports available in range $PORT_START-$PORT_END" >&2
    exit 1
}
sleep $((RANDOM % 10))
# Get a free port
MASTER_PORT=$(find_free_port)

for base_model in ${base_models[@]}; do
    for lr in ${lrs[@]}; do
        for epochs in ${epochss[@]}; do
            CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node ${gpu_count} --master_port ${MASTER_PORT} \
                train/sft_ours.py \
                --block_size=8192 \
                --per_device_train_batch_size=${micro_batch_size} \
                --per_device_eval_batch_size=${micro_batch_size} \
                --gradient_accumulation_steps=${gradient_accumulation_steps} \
                --num_train_epochs=${epochs} \
                --train_file_path=${dataset} \
                --model_name=${base_model} \
                --fsdp="full_shard auto_wrap" \
                --fsdp_config="train/llama.json" \
                --bf16=False \
                --eval_strategy="no" \
                --logging_steps=1 \
                --save_strategy="no" \
                --lr_scheduler_type cosine \
                --adam_beta1=0.9 \
                --adam_beta2=0.95 \
                --learning_rate=${lr} \
                --weight_decay=${weight_decay} \
                --output_dir="ckpts/safepath_llama" \
                --push_to_hub=${push_to_hub} \
                --hub_model_id ${model_name}_sft_sft_${epochs} \
                --save_only_model=True  \
                --gradient_checkpointing=True \
                --mode $mode
        done
    done
done
