import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from evaluation_function import monitor_process_code 
import pandas as pd
import json
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="scvi")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--gpu", type=int)
parser.add_argument("--lab", type=str)

parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")

args = parser.parse_args()

frame_work = "autogen"

model_name = args.model
tool_name = args.task
lab = args.lab
result_path = args.result_path

gpu_id = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#with open("/mnt/data00/share_data/prompt_gradient.json", "r") as f:
#    prompt_dict = eval(f.read())

root_ = f"{result_path}/{lab}/"

all_scores_ = {}

all_scores_file_path = os.path.join(root_ + f"{frame_work}/{model_name}/{frame_work}_{model_name}_scores.json")

if os.path.exists(all_scores_file_path):
    with open(all_scores_file_path, "r", encoding="utf-8") as f:
        all_scores_ = json.load(f)
else:
    with open(all_scores_file_path, "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(all_scores_file_path, "r", encoding="utf-8") as f:
        all_scores_ = json.load(f)


root_path = root_ + f"{frame_work}/{model_name}/{tool_name}/"
if os.path.exists(root_path):
    pass
else:
    print("Error! the root path does not exist!")
    raise FileNotFoundError(f"Directory does not exist: {root_path}, please run workflow first to generate results for {frame_work} combined with {model_name}.")
    #import pdb
    #pdb.set_trace()


#----------------------gpu/cpu usage
code_file_path = os.path.join(root_path, f"{tool_name}_code.py")
log_file = os.path.join(root_path, f"{tool_name}_justcode_monitor.csv")
monitor_process_code(code_file_path, log_file, gpu_id=gpu_id)

with open(log_file) as f:
    monitor_data = pd.read_csv(f)
cpu_use = monitor_data["Timestamp"][3]
cpu_ = eval(cpu_use.split(" ")[-1][0:-1]) / 100
all_scores_[tool_name]["execution"]["cpu_usage"] = cpu_
gpu_use = monitor_data["Timestamp"][7]
gpu_ = eval(gpu_use.split(" ")[-1][0:-1]) / 100
all_scores_[tool_name]["execution"]["gpu_usage"] = gpu_
#----------------------gpu/cpu usage

print(f"Finish the source usage eval of {tool_name}!")


import json

all_scores_file_path = os.path.join(root_ + f"{frame_work}/{model_name}/{frame_work}_{model_name}_scores.json")

with open(all_scores_file_path, "w", encoding="utf-8") as f:
    json.dump(all_scores_, f)


