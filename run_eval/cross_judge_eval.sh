EVAL_CONDA_ENV="agentbench"
EVAL_MODEL="genmini-2.5-pro"

PROMPT_JSON="./prompt_for_test_eval.json"
RESULT_PATH="/public/home/testdata/yangliu/agentbench/share_data/results"
GT_PATH="/public/home/testdata/yangliu/agentbench/share_data/groundtruth_result"
OUTPUT_PATH="/public/home/yangliu/workspaces/agent-benchmark-eval/just_test/cross_judge_results"
DATABASE_PATH="/public/home/testdata/yangliu/agentbench/share_data/database"
GT_PATH="/public/home/testdata/yangliu/agentbench/share_data/groundtruth_script"


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
            echo "🔧 Running cross judge for model: $model, for frame_work: $frame_work, for lab: $lab"
            echo "🔧 Using evaluation model: $EVAL_MODEL"
            
            python ./evaluation/cross_judge_eval.py --model "$model" --lab $lab --prompt_path $PROMPT_JSON --result_path $RESULT_PATH --out_path $OUTPUT_PATH --frame_work "$frame_work" --database_path $DATABASE_PATH --eval_model $EVAL_MODEL --gt_path $GT_PATH
            
        done
    done
done