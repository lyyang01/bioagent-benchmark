EVAL_CONDA_ENV="agentbench"

RESULT_JSON="/public/home/yangliu/workspaces/agent-benchmark-eval/mainresult.json"
OUTPUT_PATH="/public/home/yangliu/workspaces/agent-benchmark-eval/just_test"
W1=0.15
W2=0.15
W3=0.2
W4=0.5

models=(
    "gpt-4o"
    "gpt-4.1"
    "sonnet-3.7"
    "qwen-max"
    "genmini-2.5-pro"
    "deepseek-r1"
    "deepseek-v3"
    "grok3-beta"
)
model_ids=(
    0
    6
    3
    2
    4
    5
    1
    7
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
    echo "🔧 Running overall score computation for frame_work: $frame_work, for lab: $lab"

    for i in "${!models[@]}"
    do
        model="${models[$i]}"
        model_id="${model_ids[$i]}"

        for lab in "${lab_type[@]}"; do
            
            
            python ./evaluation/compute_overall_score.py --model "$model" --result_json $RESULT_JSON --out_path $OUTPUT_PATH --frame_work "$frame_work" --w1 $W1 --w2 $W2 --w3 $W3 --w4 $W4 --model_id $model_id
            
        done
    done
done