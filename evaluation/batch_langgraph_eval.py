#!/usr/bin/env python3
import json
import subprocess
import sys
import os
import argparse
import traceback

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fw", type=str, default="langgraph")
    parser.add_argument("--lab", type=str)
    args = parser.parse_args()

    lab = args.lab
    model = args.model
    frame_work = args.fw

    file_path = f"../results/{lab}/{frame_work}/{model}/{frame_work}_{model}_scores.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(file_path)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")
        print(f"File '{file_path}' created.")


    json_path = "../prompt_gradient.json"
    with open(json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)


    for task_key, task_info in task_data.items():
        # if args.monitor == 1 and task_key not in ['scgen-integration', 'multivi']:
        #     continue
        conda_env = task_info.get("conda_env")
        lang = task_info.get('language')
        if not conda_env:
            print(f"[Skip] Task {task_key} is missing conda_env information")
            continue

        cmd = (
            f"conda run -n '{conda_env}' python langgraph_eval.py "
            f"--model '{model}' "
            f"--tool '{task_key}' "
            f"--lang '{lang}' "
            f"--fw {frame_work} "
            f"--monitor {args.monitor} "
            f"--gpu {args.gpu} "
            f"--lab {lab}"
        )

        print(f"\n🟢 Executing command: {cmd}")

        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")       

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)    
            
        except (Exception, subprocess.CalledProcessError) as e:
            error_msg = traceback.format_exc()
            print(error_msg)
            print(f"[❌ Exception caught] {e}, skipping and continuing with the next task")
            continue            


if __name__ == "__main__":
    main()