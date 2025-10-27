
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
args = parser.parse_args()


tool_names = [args.tool]#["scanvi"]
model_name = args.model#"deepseek-v3"#"qwen-max"#deepseek-v3#"gpt-4o"
frame_work = "autogen"

all_scores_ = {}

root_path_ = f"/data/yangliu/agent-benchmark/bioagent/results/multiple-datasets/{frame_work}/{model_name}/"
#root_path_ = f"/mnt/data00/share_data/results/ablation/without_planning/{frame_work}/{model_name}/"
#root_path_ = f"/mnt/data00/share_data/results/gradient_prompt/intermediate/{frame_work}/{model_name}/"
#root_path_ = f"/mnt/data00/share_data/results/gradient_prompt/advanced/{frame_work}/{model_name}/"
#root_path_ = f"/mnt/data00/share_data/results/ablation/without_control/{frame_work}2/{model_name}/"

if os.path.exists(root_path_):
    pass
else:
    os.makedirs(root_path_)


#f = open(root_path_ + "fail_tools.txt", "a")
for tool_name in tool_names:
        all_scores_[tool_name] = {"collaboration":{}, "execution":{}, "plan":{}, "code":{}, "rag":{}}
        
        root_path = root_path_ + f"{tool_name}/"
        if os.path.exists(root_path):
            pass
        else:
            os.makedirs(root_path)

        #try:
            #def monitor_process(script_path, tool_name, model_name, log_path):
            #------------------time
        script_path = "/data/yangliu/bio-bench/llm-agent/biobench-09-freepart2.py"
        #script_path = "/home/liuyang/bio-bench/llm-agent/biobench-09-freepart2-noretrieval.py"
        #script_path = "/home/liuyang/bio-bench/llm-agent/biobench-09-freepart2-noplan.py"
        #script_path = "/home/liuyang/bio-bench/llm-agent/biobench-09-freepart2-withadvance.py"
        #script_path = "/home/liuyang/bio-bench/llm-agent/biobench-09-freepart2-nocontrol_v2.py"
        
        log_file = os.path.join(root_path, f"{tool_name}_monitor_log.csv")
        
        #change conda
        #with open("/mnt/data00/share_data/prompt_gradient.json", "r") as fr_tmp:
        #    input_dict = eval(fr_tmp.read())
        #conda_name = input_dict[tool_name]["conda_env"]

        #command = f"""
        #        conda init bash
        #        source activate base
        #        conda activate {conda_name}
        #        """

        #subprocess.Popen(f"bash -c '{command}'", shell=True)
        #activate_command = "conda activate "+conda_name
        #subprocess.Popen(activate_command, shell=True)
        #subprocess.Popen("conda deactivate", shell=True)
        #subprocess.Popen(f"conda activate "+conda_name, shell=True)


        monitor_process(script_path, tool_name, model_name, log_file)

        #with open(log_file) as f:
        #    monitor_data = pd.read_csv(f)
        #print(monitor_data)
        #time_str = monitor_data["Timestamp"][0]
        #time_ = eval(time_str.split(" ")[-2])
        #all_scores_[tool_name]["execution"]["time"] = time_
    #except:
    #    f.write(tool_name+"\n")
#f.close()
        #------------------time
        #*here to generate code.py and plan.txt from the orginal work flow