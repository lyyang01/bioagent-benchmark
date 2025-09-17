from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import traceback
import subprocess
import traceback
import re, sys
from rag import vector_store
from utils import MemoryManager
from r_env_manager import REnvironmentManager
from logger_manager import get_logger  
from config import LLMFactory
import warnings, io
from contextlib import redirect_stdout

logger = get_logger()
# llm = LLMFactory.get_instance()
# Global memory dictionary to persist the state across executions
memory_manager = MemoryManager()
# r_memory_manager = REnvironmentManager()
r_memory_manager = None

    # """Code execution function, return the result of the execution or the error log."""

@tool  
def code_exec(code: str, languange: str, current_step: str) -> str:
    """
    Execute a complete code script written in either Python or R, and return the output or error log.

    This function executes comprehensive code modules with complete program functionality. It runs Python code 
    in a namespace managed by a memory_manager and R code through an r_memory_manager. Standard output 
    and errors are captured and returned along with execution status.

    Parameters:
        code (str): Raw code to execute (multi-line supported). Do not include markdown formatting like ```python.
        languange (str): Programming language of the code ('python' or 'r').
        current_step (str): Specify which planning step is currently being executed.(using Step...)
    Returns:
        str: Execution result, either:
            - Standard output from the code, or
            - Error traceback if execution fails.
    Notes:
        - Only 'python' or 'r' are supported.
        - Designed to execute raw, standalone code scripts — not REPL commands or markdown-formatted code blocks.
        - If you want to view variable contents in python, use print(), e.g., 'print(adata)'. In R, typing a variable name alone will display its content.
    """
    try:
        global memory_manager, r_memory_manager

        output = ""  # Initialize the output to accumulate results

        stdout_capture = io.StringIO()

        logger.log_agent_output(code, "code_generate", "assistant")

        if languange.lower() == "python":
            with warnings.catch_warnings(record=True) as captured_warnings, redirect_stdout(stdout_capture):
                warnings.simplefilter("default")  # Capture all warnings  
                local_namespace = memory_manager.get_namespace()
                exec(code, {}, local_namespace)
                memory_manager.update(local_namespace)
                output += f"### Python Code Executed Successfully ###"
                stdout_output = stdout_capture.getvalue() 
                if stdout_output:
                    output += "\n"+stdout_output[:2000]
        elif languange.lower() == "r":
            if r_memory_manager is None:
                r_memory_manager = REnvironmentManager()
                            
            py_namespace = memory_manager.get_namespace()
            result, stdout_output = r_memory_manager.execute_r_code_with_output(code, py_namespace)
            updated_namespace = r_memory_manager.update_to_python(py_namespace)
            memory_manager.update(updated_namespace)
            
            if result is not None:
                py_namespace['r_result'] = result
                memory_manager.update(py_namespace)
            output += f"### R Code Executed Successfully ###"
            if stdout_output:
                output += "\n"+"\n".join(stdout_output)[:2000] 

        else:
            raise ValueError(f"Unsupported language: {languange}. Supported languages are 'python' and 'r'.")
         
        logger.log_agent_output(f'''{{output":{output},"current_step":{current_step}}}''', "code_execution_success", "user")  

        return output  

    except (Exception, BaseException) as e:
        error_message = traceback.format_exc()
        warning_message = ""
        
        if 'captured_warnings' in locals() and captured_warnings:
            warning_message = "The warning messages:\n" + "\n".join([str(warning.message) for warning in captured_warnings])
      
        if len(error_message)>10000:
            error_message=error_message[:5000]+error_message[-5000:]      
        error_message = f"Failed the {languange} code execution test:\n{error_message}\n{warning_message[:1500]}\n"
        if logger:
            logger.log_agent_output(error_message, "code_execution_error", "user")
        return error_message
    

    

@tool
def code_retrieve(query: str, current_step: str) -> str:
    """
    Retrieve the most relevant bioinformatics package name based on the input query.
    
    This tool returns relevant code snippets and usage examples based on input bioinformatics 
    tool names (such as scVI, Leiden algorithm, etc.). This helps to quickly implement 
    data analysis and processing tasks. It uses smart retrieval that only fetches new information when your queries differ significantly from previous ones.
    
    Parameters:
        query: Name of the bioinformatics tool, e.g., "scVI", "Leiden", "Decoupler", etc.
        current_step: Specify which planning step is currently being executed.(using Step...)
    Returns:
        Relevant code snippets and API usage examples, including common parameter settings 
        and best practices
        
    Examples:
        Input "scVI integration" will return code examples related to data integration using scVI
        Input "Leiden clustering" will return code examples for clustering using the Leiden algorithm
    """
    result = vector_store.similarity_search_with_relevance_scores(query, k=1)
    result = [r for r in result if r[1]>0.2]

    print("---------result to retriever_tool---------")
    # print(result)
    result = [r for r in result if r[1]>0.2]
    
    print("---------call to retriever_tool---------")
    print(f"retrieve query: {query}")
    for res in result:
        print(f"retrieve result: {res[1]} [{res[0].metadata}] \n{res[0].page_content[:200]}")
    print("---------call to retriever_tool end---------")


    if logger:
        if result:
            res = result[0]
            logger.log_agent_output(f"{{'query':{query}, 'score':{res[1]}, 'retrieved_content':{res[0].page_content},'current_step':{current_step}}}", "rag", "tool")
        else:
            logger.log_agent_output(f"{{'query':{query}, 'current_step':{current_step}}}", "rag", "tool")

    return result





