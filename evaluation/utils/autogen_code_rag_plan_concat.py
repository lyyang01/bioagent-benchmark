import re
import argparse

#fail tool may xiufu
#autogen
#sonnet-3.7: mofa, peakvi
#langgraph
#

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,default="gpt-4o")
parser.add_argument("--lab", type=str, default="main_result")
parser.add_argument("--fw", type=str, default="autogen")
parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")
parser.add_argument("--prompt_path", type=str, default="../../prompt/prompt_gradient.json")
args = parser.parse_args()

lab = args.lab
model = args.model
frame_work = args.fw

json_path = args.prompt_path
result_path = args.result_path

#model_names = ["genmini-2.5-pro", "gpt-4.1", "gpt-4o", "grok-3-beta", "sonnet-3.7", "qwen-max", "deepseek-v3", "deepseek-r1"]
model_names = [model]

with open(json_path, "r") as f:
    prompt_dict = eval(f.read())


root_path = f"{result_path}/{lab}/"
#"/data/yangliu/agent-benchmark/bioagent/results/multiple-datasets/"

tool_names = list(prompt_dict.keys())
#tool_names = ["deseq2", "resolvi"]

for model_name in model_names:
    for tool_name in tool_names:

        try:
            workflow_path = root_path + f"{frame_work}/{model_name}/{tool_name}/{tool_name}-workflow.txt"
            with open(workflow_path, 'r', encoding='utf-8') as f:
                wf = f.read()
            wf = eval(wf)

            # read and write RAG-related information
            rag_path = root_path + f"{frame_work}/{model_name}/{tool_name}/rag.txt"
            manager_all = []
            for idx, agent in enumerate(wf):
                if agent['name']=="manager":
                    #if model_name == "deepseek-v3":
                        try:
                            
                            pattern = r"```json(.*?)```"
                            tmp = agent["content"]
                            matches = re.findall(pattern, tmp, re.DOTALL)
                            tmp = '\n'.join(matches)
                            
                            if tmp == "":
                                tmp = eval(agent["content"])
                            else:
                                tmp = eval(tmp)
                            
                            if tmp["RAG Required"] == "Yes":
                                tmp2 = wf[idx+2]['content'].split("Context is: ")[-1].strip()
                                tmp["retrieved_content"] = tmp2
                                #import pdb
                                #pdb.set_trace()
                            manager_all.append(tmp)
                            #import pdb
                            #pdb.set_trace()
                            #break
                        except:
                            if "tool_calls" in agent:
                                tmp = {}
                                tmp["RAG Required"] = "Yes"
                                tmp["Current Step"] = eval(agent["tool_calls"][0]["function"]["arguments"])["message"]
                                tmp["retrieved_content"] = wf[idx+1]["content"].split("Context is: ")[-1].strip()
                                manager_all.append(tmp)
                            else:
                                pass

                    #else:
                    #    try:
                    #        assert type(eval(agent['content']))==dict
                    #        tmp = eval(agent['content'])
                    #        if tmp["RAG Required"] == "Yes":
                    #            tmp2 = wf[idx+2]['content'].split("Context is: ")[-1].strip()
                    #            tmp["retrieved_content"] = tmp2
                    #        manager_all.append(tmp)
                    #    except:
                    #        pass
            with open(rag_path, 'w', encoding='utf-8') as f:
                f.write(str(manager_all))
            
            #read and write plan
            plan_path = root_path + f"{frame_work}/{model_name}/{tool_name}/plan.txt"
            for agent in wf:
                if agent['name'] == 'planner':
                    plan_str = agent['content']
                    break
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(plan_str)
            print(f"generate {model_name}/{tool_name} plan txt")

            #read and write code
            if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
                pattern = r"```r(.*?)```"
                code_path = root_path + f"{frame_work}/{model_name}/{tool_name}/{tool_name}_code.r"
                code_list = []
                for idx, agent in enumerate(wf):
                    if agent['name'] == 'executor':
                        if "exitcode: 0" in agent['content']:
                            code_agent = wf[idx-1]
                            code_str = wf[idx-1]['content']
                            matches = re.findall(pattern, code_str, re.DOTALL)
                            tmp = '\n'.join(matches)
                            code_list.append(tmp)
                code_str = '\n'.join(code_list)
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(code_str)
                print(f"generate {model_name}/{tool_name} code R")
            else:
                pattern = r"```python(.*?)```"
                code_path = root_path + f"{frame_work}/{model_name}/{tool_name}/{tool_name}_code.py"
                code_list = []
                for idx, agent in enumerate(wf):
                    if agent['name'] == 'executor':
                        if "exitcode: 0" in agent['content']:
                            code_agent = wf[idx-1]
                            code_str = wf[idx-1]['content']
                            matches = re.findall(pattern, code_str, re.DOTALL)
                            tmp = '\n'.join(matches)
                            code_list.append(tmp)
                code_str = '\n'.join(code_list)
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(code_str)
                print(f"generate {model_name}/{tool_name} code py")
            
        except:
            print(f"*fail in {model_name}/{tool_name}!*")


        
                

