#!/bin/bash
gpu_id=0
KEYS=('scanvi') # You can add more keys as needed, the keys should match those in the JSON file

PROMPT_JSON="/data/yangliu/bioagent-benchmark/prompt/prompt_data2.json" # Path to your JSON file
LAB="test" # You can change the lab value as needed

RESULT_PATH="/data/yangliu/bioagent-benchmark/results" # Path to store results
REACT_RUN_PATH="/data/yangliu/bioagent-benchmark/src/react-code/run.py" # Path to the autogen run script

MODEL="gpt-4o" # You can change the model as needed

if [ ! -f "$PROMPT_JSON" ]; then
  echo "Error: JSON file $PROMPT_JSON does not exist."
  exit 1
fi
for KEY in "${KEYS[@]}"; do
    echo $KEY
    
    CONDA_ENV=$(python3 -c "import json, sys; data=json.load(open('$PROMPT_JSON')); print(data.get('$KEY', {}).get('conda_env', ''))")
    if [ -z "$CONDA_ENV" ]; then
        echo "Error: Unable to read 'conda_env' from $PROMPT_JSON."
        exit 1
    fi

    source activate base
    conda activate "${CONDA_ENV}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python ./run_workflow/react_run_workflow.py --task $KEY --model $MODEL --lab $LAB --prompt_path $PROMPT_JSON --result_path $RESULT_PATH --react_run_path $REACT_RUN_PATH
done