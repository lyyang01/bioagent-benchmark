#!/bin/bash
PROMPT_JSON="./prompt/prompt_for_test_eval.json"
RESULT_PATH="./logs/logs_by_workflow"
EVAL_MODEL="gpt-4o"
GT_PATH="/data/yangliu/agent-benchmark/groundtruth_script/"
GPU_ID=0
DATABASE_PATH="/data/yangliu/bioagent-benchmark/database/"

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



for model in "${models[@]}"
do
    echo "🔧 Running for model: $model"

    for lab in "${lab_type[@]}"; do
        
        #echo "$model $lab code concat"
        #python ./evaluation/utils/langgraph_code_concat.py --model "$model" --lab $lab

        #echo "$model $lab rag concat"
        #python ./evaluation/utils/langgraph_rag_concat.py --model "$model" --lab $lab

        echo "✅ Running $model $lab evaluation..."
        python ./evaluation/langgraph/batch_langgraph_eval.py --model "$model" --gpu $GPU_ID --lab $lab --prompt_path $PROMPT_JSON --result_path $RESULT_PATH --eval_model $EVAL_MODEL --gt_path $GT_PATH --database_path $DATABASE_PATH

        #python ./evaluation/compute_final_score.py --model "$model" --framework "langgraph" --lab $lab

        #bash ./output_count.sh "$lab" "langgraph" "$model"    
    done
done




