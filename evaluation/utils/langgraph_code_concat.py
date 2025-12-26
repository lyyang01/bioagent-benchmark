import json
import re
import os
import glob
import argparse

def extract_successful_code_blocks(json_data):
    """
    Extract successfully executed code blocks by finding 'code_generate' entries followed by 'code_execution_success' entries
    
    Args:
        json_data (list): Parsed JSON data containing the conversation
        
    Returns:
        str: Concatenated string of all successful code blocks
    """
    successful_code = []
    last_code_generate = None
    
    for i, entry in enumerate(json_data):
        # Store the most recent code_generate content
        if entry.get("role") == "assistant" and entry.get("name") == "code_generate":
            # Directly use content as code block (no regex extraction needed)
            content = entry.get("content", "")
            if content:
                last_code_generate = content
            else:
                last_code_generate = None
        # Also check tool and code_generate combination (adapts to different JSON structures)
        elif entry.get("role") == "tool" and entry.get("name") == "code_generate":
            content = entry.get("content", "")
            # Try to extract code from content that might be wrapped in ```python
            code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
            if code_blocks:
                last_code_generate = code_blocks[0]
            else:
                # If no ```python markers, use content directly
                last_code_generate = content
        
        # If successful execution is found and there was a previous code_generate
        elif (entry.get("role") == "user" and entry.get("name") == "code_execution_success" 
              and last_code_generate is not None):
            successful_code.append(last_code_generate)
            last_code_generate = None  # Reset to avoid duplication
    
    # Separate all successful code blocks with newlines and comments
    return ("\n\n"+"#"*30+" Next code block"+"#"*20+"\n\n").join(successful_code) if successful_code else ""


def get_latest_json(path, exclude_patterns=None):
    """
    Find the latest JSON file in the directory, excluding files matching specific patterns
    
    Args:
        path (str): Directory path to search for JSON files
        exclude_patterns (list, optional): List of regex patterns to exclude. Defaults to common exclusion patterns.
    
    Returns:
        str: Filename of the latest JSON file (without path), or None if no files found
    """
    # Default exclusion patterns if none provided
    if exclude_patterns is None:
        exclude_patterns = [
            r'_scores\.json$',    # Files ending with _scores.json
            # r'_config\.json$',     # Files ending with _config.json
            # r'_meta\.json$',       # Files ending with _meta.json
            # r'temp_.*\.json$',     # Files starting with temp_
            # r'backup_.*\.json$'    # Files starting with backup_
        ]
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(path, "*.json"))
    
    # Filter out files matching any exclusion pattern
    filtered_files = json_files.copy()
    for pattern in exclude_patterns:
        filtered_files = [f for f in filtered_files if not re.search(pattern, os.path.basename(f))]
    
    # Return the most recently modified file from the filtered list
    if filtered_files:
        return os.path.basename(max(filtered_files, key=os.path.getmtime))
    else:
        return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default="gpt-4o")
    parser.add_argument("--lab", type=str, default="main_result")
    parser.add_argument("--fw", type=str, default="langgraph")
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
        root_path = f"{result_path}/{lab}/{frame_work}/{model}/{tool}/"
        if not os.path.exists(root_path):
            print(f"Path does not exist: {root_path}")
            continue
            
        latest_json = get_latest_json(root_path)
        if not latest_json:
            print(f"No JSON files found in {root_path}")
            continue
            
        json_path = os.path.join(root_path, latest_json)
        if task_data[tool]["language"] =="R":
            file_path = os.path.join(root_path, f"{tool}_code.r")
        else:
            file_path = os.path.join(root_path, f"{tool}_code.py")
        
        try:
            # Read JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            print(f"Processing file: {json_path}")
            
            # Extract successful code blocks
            successful_code = extract_successful_code_blocks(json_data)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(successful_code)
                
            print(f"Code saved to: {file_path}")
            
        except Exception as e:
                print(f"Error processing {json_path}: {e}")
                

if __name__ == "__main__":
    main()