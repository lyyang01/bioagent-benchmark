#!/bin/bash
KEYS=('decoupler')
PROMPT_JSON="./prompt/prompt_for_test_eval.json"
RESULT_PATH="./logs/logs_by_workflow"
GPU_ID=0

models=(
    "gpt-4o"
    #"gpt-4.1"
    #"sonnet-3.7"
    #"qwen-max"
    #"genmini-2.5-pro"
    #"deepseek-r1"
    #"deepseek-v3"
    #"grok3-beta"
)

lab_type=(
    "main_result"
    # "gradient_prompt/intermediate"
    # "gradient_prompt/advanced"
    # "ablation/without_retrieve"
    # "ablation/without_planning"
)

# 检查 JSON 文件是否存在
if [ ! -f "$PROMPT_JSON" ]; then
  echo "Error: JSON file $PROMPT_JSON does not exist."
  exit 1
fi

for MODEL in "${models[@]}"; do
    echo "🔧 Running source usage eval for model: $model"
    for lab in "${lab_type[@]}"; do
        for KEY in "${KEYS[@]}"; do
            echo $KEY
            
            CONDA_ENV=$(python3 -c "import json, sys; data=json.load(open('$PROMPT_JSON')); print(data.get('$KEY', {}).get('conda_env', ''))")
            # 检查是否成功读取到环境名称
            if [ -z "$CONDA_ENV" ]; then
                echo "Error: Unable to read 'conda_env' from $PROMPT_JSON."
                exit 1
            fi
            
            source activate base
            conda activate "${CONDA_ENV}"

            python ./evaluation/autogen/autogen_eval_source_usage.py --task $KEY --model $MODEL --lab $lab --gpu $GPU_ID --result_path $RESULT_PATH
        done
    done
done