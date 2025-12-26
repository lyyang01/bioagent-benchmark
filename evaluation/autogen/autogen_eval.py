import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from evaluation_function import monitor_process_code, task_completion, round_statics, plan_score, code_score, rag_eval, task_completion_autogen, consistency_with_plan
import pandas as pd
import json
import subprocess
#"scvi","scanvi", "scanorama","harmony","cellassign", "contrastivevi", "peakvi", "poissonvi","totalvi",
#tool_names =  ["multivi"]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o") #,"gemini-2.5-pro"
parser.add_argument("--lab", type=str)

parser.add_argument("--prompt_path", type=str, default="../../prompt/prompt_gradient.json")
parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")
parser.add_argument("--eval_model", type=str, default="gpt-4o")
parser.add_argument("--gt_path", type=str, default="./groundtruth_scripts")
parser.add_argument("--database_path", type=str, default="./database")
args = parser.parse_args()


lab = args.lab
json_path = args.prompt_path
result_path = args.result_path
eval_model = args.eval_model
gt_path = args.gt_path
database_path = args.database_path
frame_work = "autogen"


#tool_names = ["wishbone"]#["scanvi"]
model_names = [args.model]

all_scores_ = {}

#root_path_ = f"/mnt/data00/share_data/results/{frame_work}/{model_name}/"
for model_name in model_names:
    print(f"********Begin eval {model_name} results************")
    root_path_ = f"{result_path}/{lab}/{frame_work}/{model_name}/"


    if os.path.exists(root_path_):
        pass
    else:
        raise FileNotFoundError(f"Directory does not exist: {root_path_}, please run workflow first to generate results for {frame_work} combined with {model_name}.")

    with open(json_path, "r") as f:
        prompt_dict = eval(f.read())

    all_scores_file_path = root_path_ + f"{frame_work}_{model_name}_scores.json"
    if os.path.exists(all_scores_file_path):
        with open(all_scores_file_path, "r", encoding="utf-8") as f:
            all_scores_ = json.load(f)

    else:
        with open(all_scores_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(all_scores_file_path, "r", encoding="utf-8") as f:
            all_scores_ = json.load(f)

    tool_names = list(prompt_dict.keys())

    f_eval_fail = open(root_path_ + f"eval_fail_tools.txt", "w")
    eval_fail = []
    
    all_keys = ["collaboration", "execution", "plan", "code", "rag", "consistency_with_plan"]
    
    for tool_name in tool_names:
        try: 
            if tool_name not in all_scores_.keys():
                all_scores_[tool_name] = {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{},"consistency_with_plan": {}}
            else:
                for key_ in all_keys:
                    if key_ not in all_scores_[tool_name].keys():
                        all_scores_[tool_name][key_] = {}
            
            root_path = root_path_ + f"{tool_name}/"
            if os.path.exists(root_path):
                pass
            else:
                print("Error! the tool path does not exist!")
                continue
                
            #------------------time
            log_file = os.path.join(root_path, f"{tool_name}_monitor_log.csv")

            with open(log_file) as f:
                monitor_data = pd.read_csv(f)
            time_str = monitor_data["Timestamp"][0]
            time_ = eval(time_str.split(" ")[-2])
            all_scores_[tool_name]["execution"]["time"] = time_
            #------------------time
            
            #----------------------plan score
            if all_scores_[tool_name]["plan"] == {}:
                plan_file_path = root_path_ + f"{tool_name}/plan.txt"

                with open(plan_file_path, "r") as f:
                    plan_str = f.read()
                plan_scores = plan_score(plan_str, tool_name, json_path, eval_model)

                all_scores_[tool_name]["plan"] = plan_scores
            else:
                pass            
            #----------------------plan score
            
            
            #----------------------code score
            if all_scores_[tool_name]["code"] == {}:
                if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_code.r"
                else:    
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_code.py"

                code_scores = code_score(code_file_path, tool_name, json_path, gt_path, eval_model)
                
                all_scores_[tool_name]["code"] = code_scores
            else:
                pass
            #----------------------code score
            
            
            #----------------------task  completion
            workflow_path = root_path_ + f"{tool_name}/{tool_name}-workflow.txt"
            with open(workflow_path) as f:
                workflow_str = f.read()
            tc_dict = task_completion_autogen(workflow_str, eval_model)
            step_number = tc_dict["step_number"]
            all_scores_[tool_name]["execution"]["task_complete_rate"] = tc_dict["task_complete_rate"]
            if frame_work != "autogen":
                all_scores_[tool_name]["success_rate"] = tc_dict["success_rate"]
            all_steps = tc_dict["all_step"]
            #----------------------task completion
            
            #----------------------round
            workflow_path = root_path_ + f"{tool_name}/{tool_name}-workflow.txt"
            with open(workflow_path, 'r', encoding='utf-8') as f:
                wf = f.read()
            workflow_list = eval(wf)

            round_list = round_statics(workflow_list, step_number)
            all_scores_[tool_name]["collaboration"] = round_list
            #----------------------round
            
            #----------------------RAG
            if all_scores_[tool_name]["rag"] == {}:
                rag_path = root_path_ + f"{tool_name}/rag.txt"

                with open(rag_path) as f:
                    rag_str = f.read()
                rag_list = eval(rag_str)
                
                if_retrieval_accuracy, retrieval_accuracy, [total_step, retrieval_step, retrieval_relate_step] = rag_eval(rag_list, tool_name, database_path, eval_model)
                try:
                    all_scores_[tool_name]["rag"]["if_retrieval_accuracy"] = eval(if_retrieval_accuracy[0:-1])/100
                    all_scores_[tool_name]["rag"]["retrieval_accuracy"] = retrieval_accuracy
                except:
                    all_scores_[tool_name]["rag"]["if_retrieval_accuracy"] = if_retrieval_accuracy/100
                    all_scores_[tool_name]["rag"]["retrieval_accuracy"] = retrieval_accuracy
            else:
                pass
            #----------------------RAG

            #----------------------consistency_with_plan
            if all_scores_[tool_name]["consistency_with_plan"] == {}:
                ## consistency with planning.
                plan_file_path = root_path_ + f"{tool_name}/plan.txt"
                with open(plan_file_path, "r") as f:
                    plan_str = f.read()
                
                if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_allcode.r"
                else:    
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_allcode.py"
                
                out = consistency_with_plan(plan_str, code_file_path, eval_model)

                all_scores_[tool_name]["consistency_with_plan"]["score"] = out["score"]
            else:
                pass
                
            #----------------------consistency_with_plan
            
            print(f"Finish the eval of {tool_name}!")
            
        except:
            print(f"***Failed on the eval of {tool_name}!!!!!")
            print(sys.exc_info())
            f_eval_fail.write(tool_name+"\n")
        
        
    f_eval_fail.close()

    #write
    all_scores_file_path = root_path_ + f"{frame_work}_{model_name}_scores.json"
    with open(all_scores_file_path, "w", encoding="utf-8") as f:
        json.dump(all_scores_, f)



