#!/bin/bash
PROMPT_JSON="./prompt/prompt_for_test_eval.json"
RESULT_PATH="./logs/logs_by_workflow"
EVAL_MODEL="gpt-4o"
GT_PATH="/data/yangliu/agent-benchmark/groundtruth_script/"
GPU_ID=0
DATABASE_PATH="/data/yangliu/bioagent-benchmark/database/"
EVAL_CONDA_ENV="agentbench"

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
source activate base
conda activate "${EVAL_CONDA_ENV}"

for model in "${models[@]}"
do
    echo "🔧 Running for model: $model"

    for lab in "${lab_type[@]}"; do

        python ./evaluation/autogen/autogen_eval.py --model "$model" --lab $lab --prompt_path $PROMPT_JSON --result_path $RESULT_PATH --eval_model $EVAL_MODEL --gt_path $GT_PATH --database_path $DATABASE_PATH
          
    done
done