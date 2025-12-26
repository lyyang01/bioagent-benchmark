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
    parser.add_argument("--fw", type=str, default="react")
    parser.add_argument("--lab", type=str)

    parser.add_argument("--prompt_path", type=str, default="../prompt/prompt_gradient.json")
    parser.add_argument("--result_path", type=str, default="../logs/logs_by_workflow")
    parser.add_argument("--eval_model", type=str, default="gpt-4o")
    parser.add_argument("--gt_path", type=str, default="./groundtruth_scripts")
    parser.add_argument("--database_path", type=str, default="./database")

    args = parser.parse_args()

    lab = args.lab
    model = args.model
    frame_work = args.fw

    result_path = args.result_path
    prompt_path = args.prompt_path
    eval_model = args.eval_model
    gt_path = args.gt_path
    database_path = args.database_path

    file_path = f"{result_path}/{lab}/{frame_work}/{model}/"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directory does not exist: {file_path}, please run workflow first to generate results for {frame_work} combined with {model}.")

    json_name = f"{frame_work}_{model}_scores.json"
    file_path = os.path.join(file_path, json_name)

    print(file_path)
    
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")
        print(f"File '{file_path}' created.")

    json_path = prompt_path
    with open(json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)


    for task_key, task_info in task_data.items():
        
        conda_env = task_info.get("conda_env")
        lang = task_info.get('language')
        if not conda_env:
            print(f"[Skip] Task {task_key} is missing conda_env information")
            continue
        

        cmd = (
            f"conda run -n '{conda_env}' python evaluation/react/react_eval.py "
            f"--model '{model}' "
            f"--tool '{task_key}' "
            f"--lang '{lang}' "
            f"--fw {frame_work} "
            f"--monitor {args.monitor} "
            f"--gpu {args.gpu} "
            f"--lab {lab} "
            f"--prompt_path {prompt_path} "
            f"--result_path {result_path} "
            f"--eval_model {eval_model} "
            f"--gt_path {gt_path} "
            f"--database_path {database_path}"
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
            # Real-time output and write to log
            for line in process.stdout:
                print(line, end="")       # Print to console

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)    

            # break
            
        except (Exception, subprocess.CalledProcessError) as e:
            error_msg = traceback.format_exc()
            print(error_msg)
            print(f"[❌ Exception caught] {e}, skipping and continuing with the next task. Eval failed for {task_key}.")
            continue

        print(f"Finish eval on {task_key}!")             


if __name__ == "__main__":
    main()