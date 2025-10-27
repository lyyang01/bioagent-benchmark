#!/bin/bash
# 初始化 Conda
# 定义 JSON 文件路径
#KEYS=('scanvi' 'decoupler' 'paga' 'destvi' 'leiden' 'mofa' 'triku' 'hotspot' 'contrastivevi' 'peakvi' 'gimvi' 'cellchat' 'sctransform') #
KEYS=('sctransform')
JSON_FILE="/data/yangliu/agent-benchmark/prompt_data2.json"
MODEL="deepseek-r1" #"grok-3-beta" #"deepseek-r1" #"gpt-4.1" #"grok-3-beta" #"grok-3-beta" #"sonnet-3.7"

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
    python ./scripts/react_run_workflow.py --tool $KEY --model $MODEL --gpu 0 --lab "main_result"
done