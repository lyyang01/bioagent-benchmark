import json
import re
import os
import glob
import argparse


def extract_successful_code_blocks(json_data):
    """
    Extract code blocks that correspond to successful code execution.
    For each 'code_execution_success' entry, find the nearest previous 'code_generate' entry
    and extract its content.
    
    Args:
        json_data (list): Parsed JSON data containing the conversation

    Returns:
        str: Concatenated string of all successful code blocks from matched code_generate entries
    """
    successful_code = []

    for idx, entry in enumerate(json_data):
        if entry.get("role") == "user" and entry.get("name") == "code_execution_success":
            # Look backwards to find the nearest 'code_generate' entry
            for prev_idx in range(idx - 1, -1, -1):
                prev_entry = json_data[prev_idx]
                if prev_entry.get("role") == "assistant" and prev_entry.get("name") == "code_generate":
                    code = prev_entry.get("content", "")
                    if code:
                        successful_code.append(code)
                    break  # Only take the nearest one

    # Join all code blocks with a divider
    return ("\n\n" + "#"*30 + " Next code block" + "#"*20 + "\n\n").join(successful_code) if successful_code else ""


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
    parser.add_argument("--model", type=str)
    parser.add_argument("--lab", type=str)
    parser.add_argument("--fw", type=str, default="react")
    args = parser.parse_args()
    
    lab = args.lab
    model = args.model
    frame_work = args.fw
    
    json_path = "/mnt/data00/share_data/prompt_gradient.json"

    with open(json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    for tool in task_data:
        root_path = f"/mnt/data00/share_data/results/{lab}/{frame_work}/{model}/{tool}/"
        if not os.path.exists(root_path):
            print(f"Path does not exist: {root_path}")
            continue
            
        latest_json = get_latest_json(root_path)
        if not latest_json:
            print(f"No JSON files found in {root_path}")
            continue
            
        json_path = os.path.join(root_path, latest_json)
        if task_data[tool]["language"] =="R":
            file_path = os.path.join(root_path, f"{tool}_code.R")
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