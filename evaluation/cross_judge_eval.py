
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import os
from evaluation_function import plan_score, code_score_cross_judge, rag_eval, consistency_with_plan
import pandas as pd
import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--lab", type=str, default="main_result")
parser.add_argument("--result_path", type=str, default="./results")
parser.add_argument("--frame_work", type=str, default="autogen")
parser.add_argument("--eval_model", type=str, default="gpt-4o")
parser.add_argument("--prompt_path", type=str, default="prompt_gradient.json")
parser.add_argument("--out_path", type=str, default="./output/")
parser.add_argument("--database_path", type=str, default="./database/")
parser.add_argument("--gt_path", type=str, default="./groundtruth_result/")

args = parser.parse_args()

model_names = [args.model]
frame_work = args.frame_work
score_model = args.eval_model
result_path = args.result_path
lab = args.lab
prompt_path = args.prompt_path
database_path = args.database_path
out_path = args.out_path
gt_path = args.gt_path

all_scores_ = {}


for model_name in model_names:
    print(f"********Begin eval {model_name} results************")
    root_path_ = f"{result_path}/{args.lab}/{frame_work}/{model_name}/"


    if os.path.exists(root_path_):
        pass
    else:
        raise Exception("The root path does not exist!")

    with open(prompt_path, "r") as f:
        prompt_dict = eval(f.read())

    current_path = f"{args.out_path}/{score_model}/"
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    #import json
    #all_scores_file_path = root_path_ + f"{frame_work}_{model_name}_scores.json"
    all_scores_file_path = f"{args.out_path}/{score_model}/" + f"{frame_work}_{model_name}_scores.json"
    if os.path.exists(all_scores_file_path):
        with open(all_scores_file_path, "r", encoding="utf-8") as f:
            all_scores_ = json.load(f)

    else:
        with open(all_scores_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(all_scores_file_path, "r", encoding="utf-8") as f:
            all_scores_ = json.load(f)

    tool_names = list(prompt_dict.keys())

    f_eval_fail = open(f"{args.out_path}/{score_model}/" + f"{frame_work}_{model_name}_eval_fail_tools.txt", "w")
    eval_fail = []
    
    all_keys = ["collaboration", "execution", "plan", "code", "rag", "consistency_with_plan"]
    for tool_name in tool_names:
        try:
            if tool_name not in all_scores_.keys():
                all_scores_[tool_name] = {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{}, "consistency_with_plan": {}}
            else:
                for key_ in all_keys:
                    if key_ not in all_scores_[tool_name].keys():
                        all_scores_[tool_name][key_] = {}
            root_path = root_path_ + f"{tool_name}/"
            if os.path.exists(root_path):
                pass
            else:
                print("Error! the root path does not exist!")
                continue
                #import pdb
                #pdb.set_trace()
            
            #----------------------plan score
            #def plan_score(plan_file_path, tool_name, output_path):
            if all_scores_[tool_name]["plan"] == {}:
                plan_file_path = root_path_ + f"{tool_name}/plan.txt"
                with open(plan_file_path, "r") as f:
                    plan_str = f.read()
                plan_scores = plan_score(plan_str, tool_name, prompt_path, eval_model=score_model)

                all_scores_[tool_name]["plan"] = plan_scores
            else:
                pass
                #print(f"Skip the plan eval of {tool_name}!")
            #----------------------plan score
            
            
            #----------------------code score
            if all_scores_[tool_name]["code"] == {}:
                if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_code.r"
                else:    
                    code_file_path = root_path_ + f"{tool_name}/{tool_name}_code.py"
                #with open(code_file_path, "r") as f:
                #    code_str = f.read()
                code_scores = code_score_cross_judge(code_file_path, tool_name, prompt_path, gt_path,eval_model=score_model)
                
                all_scores_[tool_name]["code"] = code_scores
            else:
                pass
                #print(f"Skip the code eval of {tool_name}!")
            #----------------------code score
            
            
            #----------------------RAG
            if all_scores_[tool_name]["rag"] == {}:
                rag_path = root_path_ + f"{tool_name}/rag.txt"
                with open(rag_path) as f:
                    rag_str = f.read()
                rag_list = eval(rag_str)
                #import pdb
                #pdb.set_trace()
                if_retrieval_accuracy, retrieval_accuracy, [total_step, retrieval_step, retrieval_relate_step] = rag_eval(rag_list, tool_name, database_path,eval_model=score_model)
                try:
                    all_scores_[tool_name]["rag"]["if_retrieval_accuracy"] = eval(if_retrieval_accuracy[0:-1])/100
                    all_scores_[tool_name]["rag"]["retrieval_accuracy"] = retrieval_accuracy
                except:
                    all_scores_[tool_name]["rag"]["if_retrieval_accuracy"] = if_retrieval_accuracy/100
                    all_scores_[tool_name]["rag"]["retrieval_accuracy"] = retrieval_accuracy
            else:
                pass
                #print(f"Skip the RAG eval of {tool_name}!")
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
                #with open(code_file_path, "r") as f:
                #    code_str = f.read()
                out = consistency_with_plan(plan_str, code_file_path, eval_model=score_model)

                all_scores_[tool_name]["consistency_with_plan"]["score"] = out["score"]
            else:
                pass
                #print(f"Skip the consistency_with_plan eval of {tool_name}!")
            #----------------------consistency_with_plan
            
            print(f"Finish the eval of {tool_name}!")
            
        except:
            print(f"***Failed on the eval of #{tool_name}!!!!!")
            f_eval_fail.write(tool_name+"\n")
            
    f_eval_fail.close()
    #write
    all_scores_file_path = f"{out_path}/{score_model}/" + f"{frame_work}_{model_name}_scores.json"
    
    with open(all_scores_file_path, "w", encoding="utf-8") as f:
        json.dump(all_scores_, f)



