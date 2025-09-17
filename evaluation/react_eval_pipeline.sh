#!/bin/bash


models=(
    "gpt-4o"
    "gpt-4.1"
    "sonnet-3.7"
    "qwen-max"
    "genmini-2.5-pro"
    "grok3-beta"
    "deepseek-r1"
    "deepseek-v3"
)

lab_type=(
    "main_result"
    # "gradient_prompt/intermediate"
    # "gradient_prompt/advanced"
    # "ablation/without_retrieve"
    # "ablation/without_planning"
)

gpu=0

for model in "${models[@]}"
do
    echo "🔧 Running for model: $model"

    for lab in "${lab_type[@]}"; do
        echo "🔧 Running for lab type: $lab"
        python ./batch_react_eval.py --model "$model" --monitor 1 --gpu $gpu --lab $lab
        
        echo "$model $lab code concat"
        python ./utils/react_code_contat.py --model "$model" --lab $lab

        echo "$model $lab rag concat"
        python ./utils/react_rag_contat.py --model "$model" --lab $lab

        echo "✅ Running $model $lab evaluation..."
        python ./batch_react_eval.py --model "$model" --monitor 0 --gpu $gpu --lab $lab

        python ./05_usage_compute_final_score.py --model "$model" --framework "react" --lab $lab

        bash ./output_count.sh "$lab" "react" "$model" 

    done
done





