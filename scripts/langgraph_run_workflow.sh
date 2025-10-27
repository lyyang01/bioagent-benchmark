#!/bin/bash
# 初始化 Conda
# 定义 JSON 文件路径
KEYS=('gimvi' 'scvi')
JSON_FILE="./prompt_gradient.json"
JSON_FILE="/data/yangliu/agent-benchmark/prompt_xiufu.json"

MODEL="grok3-beta" #"grok-3-beta" #"deepseek-r1" #"gpt-4.1" #"grok-3-beta" #"grok-3-beta" #"sonnet-3.7"


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
    conda activate "${CONDA_ENV}"
    python ./langgraph_run_workflow.py --tool $KEY --model $MODEL --gpu 0 --lab "main_result"
done