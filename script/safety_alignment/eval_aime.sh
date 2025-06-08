#!/bin/bash
model_name="DeepSeek-R1-Distill-Llama-8B"
guardian_model="Meta-Llama-3.1-8B-Instruct"
guardian_api_url="http://localhost:7666/v1/chat/completions"
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

# Test method
method="baseline"
echo "Running ${method} evaluation..."
CUDA_VISIBLE_DEVICES=0,1 python pred_adaptive.py \
    --model_folder ../../ckpts/${model_name}\
    --instruction_path aime25\
    --output_path ../../data/aime25/${model_name}/${method}\
    --method ${method} \
    --guardian_api_url ${guardian_api_url} \
    --guardian_model ${guardian_model} \
    --max_workers 30
CUDA_VISIBLE_DEVICES=0,1 python eval_aime.py \
    --input_path ../../data/aime25/${model_name}/${method}\
    --output_file ../../data/aime25/${model_name}/${method}_evaluation.json

method="adaptive"
echo "Running ${method} evaluation..."
CUDA_VISIBLE_DEVICES=0,1 python pred_adaptive.py \
    --model_folder ../../ckpts/${model_name}\
    --instruction_path aime25\
    --output_path ../../data/aime25/${model_name}/${method}\
    --method ${method} \
    --guardian_api_url ${guardian_api_url} \
    --guardian_model ${guardian_model} \
    --max_workers 30
CUDA_VISIBLE_DEVICES=0,1 python eval_aime.py \
    --input_path ../../data/aime25/${model_name}/${method}\
    --output_file ../../data/aime25/${model_name}/${method}_evaluation.json

method="safethink"
echo "Running ${method} evaluation..."
CUDA_VISIBLE_DEVICES=0,1 python pred_adaptive.py \
    --model_folder ../../ckpts/${model_name}\
    --instruction_path aime25\
    --output_path ../../data/aime25/${model_name}/${method}\
    --method ${method} \
    --guardian_api_url ${guardian_api_url} \
    --guardian_model ${guardian_model} \
    --max_workers 30
CUDA_VISIBLE_DEVICES=0,1 python eval_aime.py \
    --input_path ../../data/aime25/${model_name}/${method}\
    --output_file ../../data/aime25/${model_name}/${method}_evaluation.json

echo "Evaluation completed."