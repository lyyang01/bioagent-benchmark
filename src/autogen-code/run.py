
from typing import Dict, Optional, Union, List
from autogen import Agent, AssistantAgent, UserProxyAgent
import os
import autogen
from autogen.coding.jupyter import LocalJupyterServer
from autogen.coding.jupyter import JupyterCodeExecutor, JupyterConnectionInfo
from autogen.agentchat import GroupChat, GroupChatManager

import re
import math
import json
import sys
import argparse

from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from typing_extensions import Annotated
from model_config import config_list, vector_store

from types import SimpleNamespace
#from langchain.vectorstores import Chroma
#from langchain_openai import AzureOpenAIEmbeddings



parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="scvi")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--lab", type=str, default="test")
parser.add_argument("--prompt_path", type=str, default="./prompt_data.json")
parser.add_argument("--result_path", type=str, default="./bioagent/results/")

args = parser.parse_args()

if args.task in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
    KERNEL_NAME = "ir"
else:
    KERNEL_NAME = "python3"


TOOL_NAME = args.task
backend_model = args.model
prompt_path = args.prompt_path
result_path = args.result_path.rstrip("/")
config_list = config_list[backend_model]

with open(prompt_path, 'r', encoding='utf-8') as file:
    user_prompt = json.load(file)

TASK = user_prompt[args.task]["prompt_input"]["basic"]

result_type = args.lab

TASK = TASK + f" All output files need to be saved in the path '{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/agent_output'."

#save plan process
output_path = f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}"

if os.path.exists(output_path):
    pass
else:
    os.makedirs(output_path)

if os.path.exists(f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/agent_output"):
    pass
else:
    os.mkdir(f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/agent_output")

import math

llm_config = {"config_list": config_list, "seed": 42, "timeout": 7200, "temperature": 0}



# ===== Begin planner =====
#contain planner, reviewer, plan_proxy

user_message = f"""
You are a helpful assistant.
"""
user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        llm_config=llm_config,
        #max_consecutive_auto_reply=10,
        #is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        system_message=user_message
    )


user_message = f"""
You are terminator. Please reply with 'exit' to terminate the entire agent system.
"""
terminator = UserProxyAgent(
        name="terminator",
        human_input_mode="NEVER",
        llm_config=llm_config,
        #max_consecutive_auto_reply=10,
        #is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        system_message=user_message
    )

manager_message = """
You are Manager, an intelligent planning coordinator responsible for processing task plans and directing a 'coder' agent. Your two core responsibilities are:
    Step Extraction: Break down plans into executable steps and deliver the current step to the 'coder'
    RAG Decision: Determine whether knowledge retrieval (RAG) is required based on the model's autonomous processing capability for the current step

1. Plan Processing
    Analyze the full plan and decompose it into sequential steps
    Please ensure that each step you extract from the plan is complete.
2. RAG Evaluation
    Assess if the current step requires external knowledge beyond the model's base knowledge:
        IF any of these conditions apply → Trigger RAG:  
            a) Requires specialized/domain-specific knowledge or tools or packages  
            b) Needs factual verification
            c) Involves recent information (post-2023)  
            d) Contains ambiguous terms needing clarification
        If no conditions apply → No RAG needed
3. Output Format
    Always respond in this strict structure:
    {  
        "Current Step": "[Concise description of immediate task]",  
        "RAG Required": "[Yes/No]",  
        "Coder Instruction": "[Exact text to send to coder]" + ("please retrieve content" if RAG=Yes)  
    }
    - Only output information for the current step (never all steps at once)!
    - Note that if you recieve error information or questions, you can respond naturally in conversational English.

Additionally, when all steps have been completed:
Signal the end of the process by invoking the Terminator agent
   - Format your final response as:
   "All steps completed. Invoking Terminator agent to TERMINATE the process."
"""


"""
You are Manager, an intelligent planning coordinator responsible for processing task plans and directing a 'coder' agent. Your two core responsibilities are:
    Step Extraction: Break down plans into executable steps and deliver the current step to the 'coder'
    RAG Decision: Determine whether knowledge retrieval (RAG) is required based on the model's autonomous processing capability for the current step

1. Plan Processing
    Analyze the full plan and decompose it into sequential steps
    Please ensure that each step you extract from the plan is complete.
2. RAG Evaluation
    Assess if the current step requires external knowledge beyond the model's base knowledge:
        IF any of these conditions apply → Trigger RAG:  
            a) Requires specialized/domain-specific knowledge or tools or packages  
            b) Needs factual verification
            c) Involves recent information (post-2023)  
            d) Contains ambiguous terms needing clarification
        If no conditions apply → No RAG needed
3. Output Format
    Always respond in this strict structure:
    {  
        "Current Step": "[Concise description of immediate task]",  
        "RAG Required": "[Yes/No]",  
        "Coder Instruction": "[Exact text to send to coder]" + ("please retrieve content" if RAG=Yes)  
    }
    Note that if you recieve error information or questions, you can respond naturally in conversational English.

Additionally, when all steps have been completed:
Signal the end of the process by invoking the Terminator agent
   - Format your final response as:
   "All steps completed. Invoking Terminator agent to TERMINATE the process."
"""

#Please ensure that each step you extract from the plan is complete.
manager_agent = AssistantAgent(
        name="manager", 
        human_input_mode="NEVER",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message=manager_message,
    )


plan_message = f"""
You are a planner with bioinformatics expertise and need to plan the bioinformatics task based on all messages. Assuming all required packages are already installed, there is no need to consider environment configuration in your plan. Do not contain any code in your planning. Start your plan with "Plan for ..." and use "Step ..." to represent the different steps in your plan.
    """
#Note that you need to use load data instead of importing libraries as your first step.
#if you think you do not have enough information, you can ask 'user_proxy' again.
plan_agent = AssistantAgent(
        name="planner", 
        human_input_mode="NEVER",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message=plan_message
    )

#You are responsible for planning and coding with bioinformatics expertise. For planning, you need to plan the bioinformatics task based on all messages. When making plans, please proceed with the planning according to the sequence of data preprocessing, model building and training, and post-analysis. For coding, you need to write code for each step in the plan.

code_message = """
You are Coder, an AI bioinformatics specialist focused on generating executable code solutions for biological data analysis with python or R kernels. Your primary mission is to translate analytical requirements into functional code with biological relevance and correct the error from 'executor'. If necessary, please use GPU settings in your code to accelerate the training speed.
- Core Responsibilities:
(1) Precision Coding: Produce accurate, optimized code for biological data processing/analysis
(2) Domain Awareness: Prioritize bioinformatics-specific libraries/tools (Biopython, DESeq2, GATK, etc.)
(3) Explanation Integration: Embed contextual comments explaining biological relevance

- Constraints:
(1) Always include code blocks in the reply (i.e., ```python #your_code ``` or ```r #your_code ```). For document-related steps, you can write a txt file.
(2) Never show placeholder/pseudocode - always executable
(3) Prefer domain-specific packages over generic alternatives
(4) Highlight biological assumptions in code comments

Note that instead of writing all codes once, you need to write code step by step based on the information from 'manager' to finish the user task.
Your reply must be not empty!
"""

f"""
You are a coder with bioinformatical expertise and need to write code based on all messages. If necessary, please use GPU settings in your code to accelerate the training speed. Always include python codes in your reply.
"""
#Assuming you have a jupyter kernel, so you can continuously execute code cell by cell and you can directly use existing variables in previous codes in current writing.
#if you think you do not have enough information, you can ask 'user_proxy' again.
code_agent = AssistantAgent(
        name="coder", 
        llm_config={"config_list": config_list, "seed": 42, "timeout": 1000, "temperature": 0},
        human_input_mode="NEVER",
        #is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") if x.get("content", "") is not None else False,
        system_message=code_message
    )

executor_messages = f"""
        You are an executor with a python kernel, responsible for executing the code from 'coder'.
        """
    #server = LocalJupyterServer(ip="localhost", port=8888)
#server = JupyterConnectionInfo(host="127.0.0.1", use_https=False, port=8888, token=jupytertoken)
server = LocalJupyterServer()
executor_agent = AssistantAgent(
        name="executor",
        code_execution_config={
            "executor": "ipython-embedded",
            "ipython-embedded": {"output_dir": f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/agent_output","timeout": 7200},
            #"executor": JupyterCodeExecutor(server, timeout=7200, output_dir=f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/agent_output", kernel_name=KERNEL_NAME)
        },
        llm_config=llm_config,
        system_message=executor_messages
    )



def retrieve_content(
    message: Annotated[
        str,
        "Refined message which keeps the original meaning of the current step and can be used to retrieve content for code generation.",
        #f"Select one of tools in {tools_list_str} for the current step. If no tools are suitable in the tool list, reply with 'empty'. Note that reply with 'empty' for data preprocessing steps!",
        #"Extract specific task from the top line 'Refined Step ...' or 'Step ...' or 'Plan for ...' that can be used to retrieve content for code generation.",
    ]
) -> str:
    
    #results = vector_store.similarity_search(message, k=1)
    results = vector_store.similarity_search_with_relevance_scores(message, k=1)
    #import pdb
    #pdb.set_trace()
    #results = [r for r in results if r[1]>0.2]
    #import pdb
    #pdb.set_trace()
    try:
        retrieval_text = results[0][0].page_content
    except:
        retrieval_text = "No relevant content was found in the knowledge base. Please write code based on your own knowledge."
    
    PROMPT_CODE = f"""You're a retrieve augmented coding assistant. You answer user's questions or make modifications based on your own knowledge and the context provided by the user. It should be noted that not all content in the Context must be included in your answer. If the current context does not contain the knowledge you need to answer, then ignore the context and answer using only your own knowledge. For code generation, you must obey the following rules:
Rule 1. You MUST NOT install any packages because all the packages needed are already installed.
Rule 2. You must follow the formats below to write your code:
```python
# your code
```
or
```r
# your code
```r

User's question is: {message}

Context is: {retrieval_text}
"""
    return PROMPT_CODE


for caller in [code_agent]:
    d_retrieve_content = caller.register_for_llm(
        #name="retrievar",
        description="invoke the tool only if RAG Required is Yes.", #you retrieve content for code generation.
        api_style="tool"#"tool"#"function"
)(retrieve_content)

for executor in [code_agent]:
    executor.register_for_execution()(d_retrieve_content)


def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat, last_last_speaker: str, chat_round:int):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        return plan_agent
    if last_speaker is plan_agent:
        #import pdb
        #pdb.set_trace()
        #if last_last_speaker == 'user_proxy2':
        #    return user_proxy2
        #else:
        return manager_agent
    
    if last_speaker is manager_agent:
        #if last_last_speaker == 'user_proxy2':
        #    return user_proxy2
        if messages[-1]["content"].startswith("..."):
            return terminator
        if "TERMINATE" in messages[-1]["content"]:
            return terminator
        else:
            
            return code_agent
    if last_speaker is code_agent:
        #if last_last_speaker == 'user_proxy2':
        #    return user_proxy2
        #exit the whole agent system

        if "TERMINATE" in messages[-1]["content"]:
            return terminator
        #import pdb
        #pdb.set_trace()
        #if (messages[-1]["role"] == "function") or (messages[-1]["content"]==""):
        if (messages[-1]["role"] == "tool") or (messages[-1]["content"]=="") or ("tool_calls" in messages[-1].keys()):
            return code_agent
        #elif "```python" in messages[-1]["content"]:
        else:
            if ("```python" not in messages[-1]["content"]) and ("```r" not in messages[-1]["content"]) and ("```R" not in messages[-1]["content"]):
                return manager_agent
            else:
                return executor_agent
        

    if last_speaker is executor_agent:
        #if last_last_speaker == 'user_proxy2':
        #    return user_proxy2
        if "exitcode: 0" in messages[-1]["content"]:
            return manager_agent
        
        else:
            return code_agent
    

MAX_ROUND = 100
groupchat = GroupChat(agents=[user_proxy, terminator, manager_agent, plan_agent, code_agent, executor_agent], messages=[], max_round=MAX_ROUND,speaker_selection_method=custom_speaker_selection_func)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

if len(groupchat.messages) == 0:
    chat_history = user_proxy.initiate_chat(manager, message=TASK)

elif len(groupchat.messages) == MAX_ROUND:  
    chat_history = user_proxy.send(manager, message="exit",)


tmp = f"{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/"
if os.path.exists(tmp):
    pass
else:
    os.makedirs(tmp)

f = open(f'{result_path}/{result_type}/autogen/{backend_model}/{TOOL_NAME}/{TOOL_NAME}-workflow.txt', 'w')
f.write(str(groupchat.messages))
f.close()