
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.evaluation_function import monitor_process
import pandas as pd
import json
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="scvi")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--lab", type=str, default="")
parser.add_argument("--prompt_path", type=str, default="./prompt_data.json")
parser.add_argument("--result_path", type=str, default="./bioagent/results/")
parser.add_argument("--langgraph_run_path", type=str, default="./src/langgraph-code/run.py")
#parser.add_argument("--gpu", type=str, default="")


args = parser.parse_args()


tool_names = [args.task]#["scanvi"]
model_name = args.model#"deepseek-v3"#"qwen-max"#deepseek-v3#"gpt-4o"
result_type = args.lab
prompt_path = args.prompt_path
result_path = args.result_path.rstrip("/")
langgraph_code_path = args.langgraph_run_path



frame_work = "langgraph"

all_scores_ = {}

root_path_ = f"{result_path}/{result_type}/{frame_work}/{model_name}/"

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

       
        script_path = langgraph_code_path
        
        log_file = os.path.join(root_path, f"{tool_name}_monitor_log.csv")
        

        #monitor_process(script_path, result_type, tool_name, model_name, log_file)
        monitor_process(script_path, tool_name, model_name, result_type, prompt_path, result_path, log_file)

