import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_function import monitor_process_code, task_completion_langgraph, round_statics_m2, plan_score, code_score, rag_eval, consistency_with_plan_lang
import pandas as pd
import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.langgraph_code_concat import get_latest_json
# from langgraph_code_concat import get_latest_json

def time_eval(all_scores, root_path, tool_name):

    log_file = os.path.join(root_path, f"{tool_name}_monitor_log.csv")

    print(log_file)
    
    #------------------time
    with open(log_file) as f:
        monitor_data = pd.read_csv(f)
    #print(monitor_data)
    time_str = monitor_data["Timestamp"][0]
    time_ = eval(time_str.split(" ")[-2])
    all_scores["execution"]["time"] = time_
    #------------------time
    return all_scores


def gpu_eval(all_scores,root_path,tool_name,gpu_id):
    #----------------------gpu/cpu usage
    #def monitor_process_code(code_file_path, log_file):
    # code_file_path = os.path.join(root_path, f"{tool_name}_code.py")
    # log_file = os.path.join(root_path, f"code_{tool_name}.csv")
    
    code_file_path = os.path.join(root_path, f"{tool_name}_code.py")
    log_file = os.path.join(root_path, f"{tool_name}_justcode_monitor.csv")

    monitor_process_code(code_file_path, log_file, gpu_id)

    with open(log_file) as f:
        monitor_data = pd.read_csv(f)
    cpu_use = monitor_data["Timestamp"][3]
    cpu_ = eval(cpu_use.split(" ")[-1][0:-1]) / 100
    all_scores["execution"]["cpu_usage"] = cpu_
    gpu_use = monitor_data["Timestamp"][7]
    gpu_ = eval(gpu_use.split(" ")[-1][0:-1]) / 100
    all_scores["execution"]["gpu_usage"] = gpu_
    #----------------------gpu/cpu usage    
    return all_scores 


def plan_eval(all_scores, json_data, tool_name, prompt_path, eval_model):
    # plan_file_path = f"../results/{frame_work}/{model_name}/{tool_name}/plan.txt"
    # with open(plan_file_path, "r") as f:
    #     plan_str = f.read()
    plan_str=""    
    for i, entry in enumerate(json_data):

        if entry.get("role") == "assistant" and entry.get("name") == "planner":

            plan_str = entry.get("content", "")
            break

    if plan_str:      
        plan_scores = plan_score(plan_str, tool_name, prompt_path, eval_model)
        all_scores["plan"] = plan_scores
    else:
        print("No planning find!")
    return all_scores , plan_str
    # return plan_str

def code_eval(all_scores,root_path,tool_name,lang, prompt_path, gt_path, eval_model):
    # code_file_path = f"../results/{frame_work}/{model_name}/{tool_name}/{tool_name}_code.py"
    if lang=="R":
        code_file_path = os.path.join(root_path, f"{tool_name}_code.r")
    else:
        code_file_path = os.path.join(root_path, f"{tool_name}_code.py")

    if not os.path.exists(code_file_path) or os.path.getsize(code_file_path) == 0:
        all_scores["code"]={}
    else:
        code_scores = code_score(code_file_path, tool_name, prompt_path, gt_path, eval_model)

        all_scores["code"] = code_scores

    return all_scores 


def task_eval(all_scores,json_data):
    workflow_str=str(json_data)
    # tc_dict = task_completion(workflow_str)
    tc_dict = task_completion_langgraph(workflow_str)
    step_number = tc_dict["step_number"]
    all_scores["execution"]["task_complete_rate"] = tc_dict["task_complete_rate"]
    all_scores ["execution"]["success_rate"] = tc_dict["success_rate"]
    return all_scores, step_number

def round_eval(all_scores,json_data, step_number):
    workflow_list = json_data
    round_list = round_statics_m2(workflow_list, step_number)
    all_scores["collaboration"] = round_list
    return all_scores 

def ragtool_eval(all_scores, root_path, database_path, eval_model):
    rag_path = os.path.join(root_path, "rag.txt")
    with open(rag_path) as f:
        rag_str = f.read()
    rag_list = eval(rag_str)
    if_retrieval_accuracy, retrieval_accuracy, [total_step, retrieval_step, retrieval_relate_step] = rag_eval(rag_list, tool_name, database_path, eval_model)
    all_scores["rag"]["if_retrieval_accuracy"] = eval(if_retrieval_accuracy[0:-1])/100
    all_scores["rag"]["retrieval_accuracy"] = retrieval_accuracy
    return all_scores


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--tool", type=str)
parser.add_argument("--lang", type=str)
parser.add_argument("--fw", type=str, default="langgraph")
parser.add_argument("--monitor", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lab", type=str)
parser.add_argument("--prompt_path", type=str, default="../../prompt/prompt_gradient.json")
parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")
parser.add_argument("--eval_model", type=str, default="gpt-4o")
parser.add_argument("--gt_path", type=str, default="./groundtruth_scripts")
parser.add_argument("--database_path", type=str, default="./database")

args = parser.parse_args()

json_path = args.prompt_path
result_path = args.result_path
eval_model = args.eval_model
gt_path = args.gt_path
database_path = args.database_path

#log_dir = "logs"
#os.makedirs(log_dir, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    task_data = json.load(f)

lab = args.lab
frame_work = args.fw
tool_name = args.tool
model_name = args.model
gpu_id = args.gpu

# >>>>>>>> score
all_scores_file_path = f"{result_path}/{lab}/{frame_work}/{model_name}/{frame_work}_{model_name}_scores.json"

with open(all_scores_file_path, "r", encoding="utf-8") as f:
    content = f.read().strip()
    all_scores = json.loads(content) if content else {}


root_path = f"{result_path}/{lab}/{frame_work}/{model_name}/{tool_name}/"
if os.path.exists(root_path):
    pass
else:
    print(f"No results for {tool_name} under {frame_work} and {model_name}. Creating directory: {root_path}...")
    os.makedirs(root_path)
    os.makedirs(os.path.join(root_path,'agent_output'))


scores_file_path = os.path.join(root_path, f"{tool_name}_scores.json")
if os.path.exists(scores_file_path):
    with open(scores_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        all_score = json.loads(content) if content else {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{}}
else:
    all_score = {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{}}


workflow_json_path = os.path.join(root_path, get_latest_json(root_path))
with open(workflow_json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

#time
all_score = time_eval(all_score, root_path, tool_name=tool_name)

#plan
all_score, plan_str = plan_eval(all_score, json_data, tool_name, json_path, eval_model)

#code
all_score = code_eval(all_score,root_path,tool_name, args.lang, json_path, gt_path, eval_model)

all_score, step_number = task_eval(all_score,json_data)

all_score = round_eval(all_score, json_data, step_number)

all_score = ragtool_eval(all_score, root_path, database_path, eval_model)

all_score["consistency_with_plan"] = consistency_with_plan_lang(json_data, plan_str, eval_model)

#if use gpu_eval please uncomment below
#if args.lang=='python':
#    all_score = gpu_eval(all_score, root_path, tool_name, gpu_id)

all_scores[tool_name] = all_score


with open(scores_file_path, "w", encoding="utf-8") as f:
    json.dump(all_score, f, indent=4)
f.close()

with open(all_scores_file_path, "w", encoding="utf-8") as f:
    json.dump(all_scores, f)
f.close()

    