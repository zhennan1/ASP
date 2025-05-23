#!/bin/bash
model_name="safepath_qwen"

uid="$(date +%Y%m%d_%H%M%S)"
push_to_hub=False
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


cd poison/evaluation  

CUDA_VISIBLE_DEVICES=0,1 python pred.py \
    --model_folder ../../ckpts/${model_name}\
    --output_path ../../data/poison_dataset/${model_name}/baseline\
    --method baseline

CUDA_VISIBLE_DEVICES=0,1 python eval_sentiment.py \
    --input_path ../../data/poison_dataset/${model_name}/baseline\
    --method baseline