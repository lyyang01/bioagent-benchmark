import json
import os
from langgraph_code_concat import get_latest_json
import argparse


def extract_structured_steps(input_json):
    """
    Extract planner_steps and rag_tool content and convert them into a structured format
    
    Args:
        input_json (list): Input JSON data
        
    Returns:
        list: Structured JSON data
    """
    structured_steps = []
    
    # Extract all planner_steps and their indices
    step_indices = []
    for i, item in enumerate(input_json):
        if item["role"] == "assistant" and item["name"] == "planner_steps":
            step_indices.append((i, item["content"]))
    
    # Process each step
    for i, (step_idx, step) in enumerate(step_indices):
        # Determine the current step’s range
        next_step_idx = step_indices[i+1][0] if i < len(step_indices) - 1 else len(input_json)
        
        # Find all rag_tool within the step range
        rag_results = []
        has_rag = False
        
        for j in range(step_idx + 1, next_step_idx):
            if input_json[j]["role"] == "tool" and input_json[j]["name"] == "rag":
                has_rag = True
                content = input_json[j]["content"]
                
                # Try to parse rag_tool content
                rag_result = {}
                try:
                    # Try parsing as JSON
                    data = json.loads(content.replace("'", "\""))
                    for key in ["query", "score", "retrieved_content"]:
                        if key in data:
                            rag_result[key] = data[key]
                except:
                    # If parsing fails, use string-based processing
                    for key in ["query", "score", "retrieved_content"]:
                        key_str = f"'{key}':"
                        if key_str in content:
                            parts = content.split(key_str, 1)
                            if len(parts) > 1:
                                value = parts[1].split(",", 1)[0].strip() if key != "retrieved_content" else parts[1].strip()
                                
                                # Clean and process
                                if key == "retrieved_content" and value.endswith("}"):
                                    value = value[:-1].strip()
                                
                                if value.startswith("'") and value.endswith("'"):
                                    value = value[1:-1]
                                
                                # Try converting score to numeric
                                if key == "score":
                                    try:
                                        value = float(value)
                                    except:
                                        pass
                                
                                rag_result[key] = value
                
                # If any content was extracted, add to results
                if rag_result:
                    rag_results.append(rag_result)
        
        # Create structured step
        structured_step = {
            "Current Step": step,
            "RAG Required": "Yes" if has_rag else "No"
        }
        
        # If RAG results exist, add them to the step
        if rag_results:
            structured_step["rag_results"] = rag_results
        
        structured_steps.append(structured_step)
    
    return structured_steps


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lab", type=str)
    parser.add_argument("--fw", type=str, default="langgraph")
    args = parser.parse_args()

    lab = args.lab
    model = args.model
    frame_work = args.fw
    
    json_path = "../prompt_gradient.json"

    with open(json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    for tool in task_data:
        root_path = f"../results/{lab}/{frame_work}/{model}/{tool}/"
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
            structured_data = extract_structured_steps(json_data)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
                

if __name__ == "__main__":
    main()