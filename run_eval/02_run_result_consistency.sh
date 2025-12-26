EVAL_CONDA_ENV="agentbench"
PROMPT_JSON="/public/home/testdata/yangliu/agentbench/share_data/prompt_gradient.json"
RESULT_PATH="/public/home/testdata/yangliu/agentbench/share_data/results"
GT_PATH="/public/home/testdata/yangliu/agentbench/share_data/groundtruth_result"
OUTPUT_PATH="/consistency_resultspublic/home/yangliu/workspaces/agent-benchmark-eval/just_test"


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
frame_works=(
    "autogen"
    #"langgraph"
    #"react"
)

source activate base
conda activate "${EVAL_CONDA_ENV}"

for frame_work in "${frame_works[@]}"
do

    for model in "${models[@]}"
    do

        for lab in "${lab_type[@]}"; do
            echo "🔧 Running consistency computation for model: $model, for frame_work: $frame_work, for lab: $lab"
            
            python ./evaluation/compute_result_consistency.py --model "$model" --lab $lab --prompt_path $PROMPT_JSON --result_path $RESULT_PATH --gt_path $GT_PATH --out_path $OUTPUT_PATH --frame_work "$frame_work"
            
        done
    done
done