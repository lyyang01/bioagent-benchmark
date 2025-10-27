
import os
from evaluation_function import monitor_process, monitor_process_code, task_completion, round_statics, plan_score, code_score, rag_eval
import pandas as pd
import json
import subprocess
#"scvi","scanvi", "scanorama","harmony","cellassign", "contrastivevi", "peakvi", "poissonvi","totalvi",
#tool_names =  ["multivi"]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tool", type=str, default="scvi")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--lab", type=str, default="")
parser.add_argument("--gpu", type=str, default="")


args = parser.parse_args()


tool_names = [args.tool]#["scanvi"]
model_name = args.model#"deepseek-v3"#"qwen-max"#deepseek-v3#"gpt-4o"
lab_name = args.lab
gpu_id = args.gpu

frame_work = "langgraph"

all_scores_ = {}

root_path_ = f"/data/yangliu/agent-benchmark/bioagent/results/{lab_name}/{frame_work}/{model_name}/"

if os.path.exists(root_path_):
    pass
else:
    os.makedirs(root_path_)


for tool_name in tool_names:
    
        all_scores_[tool_name] = {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{}}
        
        root_path = root_path_ + f"{tool_name}/"
        if os.path.exists(root_path):
            pass
        else:
            os.makedirs(root_path)

       
        script_path = "/langgraph-code/run.py"
        
        log_file = os.path.join(root_path, f"{tool_name}_monitor_log.csv")
        

        monitor_process(script_path, lab_name, tool_name, model_name, log_file, gpu_id)

