#!/bin/bash

KEYS=('scanvi') 
JSON_FILE="./prompt_gradient.json"

MODEL="grok-3-beta" #"qwen-max" #"grok-3-beta" #"deepseek-r1" #"gpt-4.1" #"grok-3-beta" #"grok-3-beta" #"sonnet-3.7"

if [ ! -f "$JSON_FILE" ]; then
  echo "Error: JSON file $JSON_FILE does not exist."
  exit 1
fi
for KEY in "${KEYS[@]}"; do
    echo $KEY
    CONDA_ENV=$(python3 -c "import json, sys; data=json.load(open('$JSON_FILE')); print(data.get('$KEY', {}).get('conda_env', ''))")
    if [ -z "$CONDA_ENV" ]; then
        echo "Error: Unable to read 'conda_env' from $JSON_FILE."
        exit 1
    fi
    
    source activate base
    conda activate "${CONDA_ENV}"
    CUDA_VISIBLE_DEVICES=0 python ./scripts/autogen_run_workflow.py --tool $KEY --model $MODEL
done