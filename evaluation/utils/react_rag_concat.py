import os
import re
import ast
from langchain_openai import AzureChatOpenAI
from langgraph_code_concat import get_latest_json
import json
import argparse
import traceback
from itertools import groupby
from operator import itemgetter

os.environ["AZURE_OPENAI_API_VERSION"] = "XXX"
os.environ["AZURE_OPENAI_ENDPOINT"] = "XXX"
os.environ["AZURE_OPENAI_API_KEY"] = "XXX"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "XXX"

llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    temperature=0, 
    streaming=True,
    # max_retries=2,
)


def extract_structured_steps(json_data):
    # Extract planning steps
    plan_message = next((
        msg["content"] for msg in json_data 
        if isinstance(msg, dict) and msg.get("role") == "assistant" 
        and msg.get("name") == "react_agent" and isinstance(msg.get("content"), str)
        and any(kw.lower() in msg["content"].lower() for kw in ["Plan for", "Step 1", "Step 2"])
    ), None)
    
    # Get structured steps from LLM
    prompt = f'Extract the planning steps as: ```json ["Step 1 ...", "Step 2 ...", ...] ``` From: {plan_message}'
    result = llm.invoke(prompt)
    
    try:
        structured_steps = json.loads(result.content.split("```json")[1].split("```")[0])
    except Exception as e:
        print(traceback.format_exc())
        raise ValueError("Failed to parse structured steps from LLM response") from e
    
    # Extract RAG information
    rag_entries = []
    for msg in json_data:
        if (isinstance(msg, dict) and msg.get("role") == "tool" 
            and msg.get("name") == "rag" and isinstance(msg.get("content"), str)):
            
            content_str = msg["content"]
            try:
                # Try using ast.literal_eval first
                try:
                    content_dict = ast.literal_eval(content_str)
                except (SyntaxError, ValueError):
                    # Fallback to regex extraction
                    content_dict = {}

                    patterns = {
                        "query": r"'query'\s*:\s*(.+?)(?=\s*,\s*'(?:score|current_step|retrieved_content)'\s*:|$)",
                        "score": r"'score'\s*:\s*(.+?)(?=\s*,\s*'(?:query|current_step|retrieved_content)'\s*:|$)",
                        "retrieved_content": r"'retrieved_content'\s*:\s*(.+?)(?=\s*,\s*'(?:query|score|current_step)'\s*:|$)",
                        "current_step": r"'current_step'\s*:\s*(.+?)(?=\s*,\s*'(?:query|score|retrieved_content)'\s*:|$|\})"
                    }

                    for key, pattern in patterns.items():
                        match = re.search(pattern, content_str, re.DOTALL)
                        if match:
                            value = match.group(1).strip()
                            if key == "score":
                                try:
                                    value = float(value)
                                except ValueError:
                                    continue
                            content_dict[key] = value

                # Get step number
                step_str = str(content_dict.get("current_step", ""))
                match = re.search(r"Step\s*(\d+)", step_str)
                if not match:
                    continue
                    
                step = int(match.group(1))
                rag_entries.append({
                    "step": step,
                    "query": content_dict.get("query"),
                    "score": content_dict.get("score"),
                    "retrieved_content": content_dict.get("retrieved_content")
                })
                
            except Exception as e:
                print(traceback.format_exc())
                print(f"⚠️ Error parsing RAG content: {e}")
                continue
    
    # Merge planning steps with RAG information
    merged_steps = []
    for step_text in structured_steps:
        match = re.match(r"Step\s*(\d+)", step_text)
        step_number = int(match.group(1)) if match else None
        
        step_entry = {
            "Current Step": step_text,
            "RAG Required": "No"
        }
        
        if step_number:
            matching_rags = [r for r in rag_entries if r["step"] == step_number]
            if matching_rags:
                step_entry["RAG Required"] = "Yes"
                step_entry["rag_results"] = []
                
                for rag in matching_rags:
                    rag_result = {"query": rag["query"]}
                    if rag.get("score") not in [None, ""]:
                        rag_result["score"] = rag["score"]
                    if rag.get("retrieved_content") not in [None, ""]:
                        rag_result["retrieved_content"] = rag["retrieved_content"]
                    step_entry["rag_results"].append(rag_result)
                step_entry["rag_results"] = list({item["query"]: item for item in step_entry["rag_results"]}.values())
        merged_steps.append(step_entry)
    

    return merged_steps

def extract_structured_steps_noplan(json_data):
    
    # Extract RAG information
    rag_entries = []
    for msg in json_data:
        if (isinstance(msg, dict) and msg.get("role") == "tool" 
            and msg.get("name") == "rag" and isinstance(msg.get("content"), str)):
            
            content_str = msg["content"]
            try:
                # Try using ast.literal_eval first
                try:
                    content_dict = ast.literal_eval(content_str)
                except (SyntaxError, ValueError):
                    # Fallback to regex extraction
                    content_dict = {}


                    patterns = {
                        "query": r"'query'\s*:\s*(.+?)(?=\s*,\s*'(?:score|current_step|retrieved_content)'\s*:|$)",
                        "score": r"'score'\s*:\s*(.+?)(?=\s*,\s*'(?:query|current_step|retrieved_content)'\s*:|$)",
                        "retrieved_content": r"'retrieved_content'\s*:\s*(.+?)(?=\s*,\s*'(?:query|score|current_step)'\s*:|$)",
                        "current_step": r"'current_step'\s*:\s*(.+?)(?=\s*,\s*'(?:query|score|retrieved_content)'\s*:|$|\})"
                    }

                    for key, pattern in patterns.items():
                        match = re.search(pattern, content_str, re.DOTALL)
                        if match:
                            value = match.group(1).strip()
                            if key == "score":
                                try:
                                    value = float(value)
                                except ValueError:
                                    continue
                            content_dict[key] = value

                # Get step number
                step_str = str(content_dict.get("current_step", ""))
                # Get step number
                step_str = str(content_dict.get("current_step", ""))
                match = re.search(r"Step\s*(\d+)", step_str, re.IGNORECASE)
                if not match:
                    continue
                    
                step = int(match.group(1))

                rag_entries.append({
                    "step": step,
                    "current_step": step_str,
                    "query": content_dict.get("query"),
                    "score": content_dict.get("score"),
                    "retrieved_content": content_dict.get("retrieved_content")
                })
                
            except Exception as e:
                print(traceback.format_exc())
                print(f"⚠️ Error parsing RAG content: {e}")
                continue
    
    # Merge planning steps with RAG information
    merged_steps = []

    rag_entries_sorted = sorted(rag_entries, key=itemgetter("step"))
    grouped_by_step = {
        step: list(items)
        for step, items in groupby(rag_entries_sorted, key=itemgetter("step"))
    }


    for step, step_rag_list in grouped_by_step.items():
        
        step_entry = {
            "Current Step": step_rag_list[0]["current_step"],
            "RAG Required": "Yes",
            "rag_results": []
        }
        for rag in step_rag_list:
            rag_result = {"query": rag["query"]}
            if rag.get("score") not in [None, ""]:
                rag_result["score"] = rag["score"]
            if rag.get("retrieved_content") not in [None, ""]:
                rag_result["retrieved_content"] = rag["retrieved_content"]
            step_entry["rag_results"].append(rag_result)
        step_entry["rag_results"] = list({item["query"]: item for item in step_entry["rag_results"]}.values())
        merged_steps.append(step_entry)
        return merged_steps


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o",type=str)
    parser.add_argument("--lab", type=str, default="main_result")
    parser.add_argument("--fw", type=str, default="react")
    parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")
    parser.add_argument("--prompt_path", type=str, default="../../prompt/prompt_gradient.json")

    args = parser.parse_args()

    lab = args.lab
    model = args.model
    frame_work = args.fw

    json_path = args.prompt_path
    result_path = args.result_path


    with open(json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    for tool in task_data:
        # if tool not in ["celltypist"]:
        #     continue
        print(tool)
        # if tool != "leiden":
        #     continue
        root_path = f"{result_path}/{lab}/{frame_work}/{model}/{tool}/"
        if not os.path.exists(root_path):
            print(f"Path does not exist: {root_path}")
            continue
            
        latest_json = get_latest_json(root_path)
        if not latest_json:
            print(f"No JSON files found in {root_path}")
            continue
            
        json_path = os.path.join(root_path, latest_json)
        output_file_path = os.path.join(root_path, f"rag.txt")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            print(f"Processing file: {json_path}")
            if lab in ["ablation/without_planning"]:
                structured_data=extract_structured_steps_noplan(json_data)
            else:
                structured_data=extract_structured_steps(json_data)

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    main()
    
