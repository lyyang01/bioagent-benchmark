import json
from typing import Literal
from rich.console import Console
from rich.syntax import Syntax
from graph import GraphState
from agent_chains import CodeGenerator, PlanGenerator , ToolCaller
from rag import vector_store
import traceback
import warnings, io
from contextlib import redirect_stdout
from langchain.tools import tool
from model_config import LLMFactory
from logger_manager import get_logger  
from utils import MemoryManager
from r_env_manager import REnvironmentManager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

console = Console()
retrieve_flag = 0

# Initialize model and code generator
llm = LLMFactory.get_instance()
code_gen_chain = CodeGenerator(llm)._create_chain()
plan_chain = PlanGenerator(llm)._create_chain()

logger = get_logger()
memory_manager = MemoryManager()
# r_memory_manager = REnvironmentManager()
r_memory_manager = None

## node
def code_generate(state: GraphState):
    print("---Generate code solution---")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]    
        
    if error:
        messages.append(HumanMessage(content="Now, try again. Based on the previous error message, revise the part of the code that caused the issue. \
                            Do not mechanically repeat the same line or block of code that previously resulted in an error. \
                            Instead, thoughtfully modify the original code based on the cause of the error, rather than simply appending a workaround."))
    

    code_solution = code_gen_chain.invoke({"current_plan":state["current_plan"],"messages": str(messages)})   

    code_solution = code_solution.content

    messages.append(AIMessage(content=code_solution))
    iterations = iterations + 1
    
    # syntax = Syntax(code_solution.code, "python", theme="monokai", line_numbers=True, word_wrap=True)
    # console.print(syntax)
    
    if logger:
        logger.log_agent_output(code_solution , "code_generate", "assistant")
    
    return {"code_solution": code_solution, "messages": messages, "iterations": iterations}


def code_execution(state: GraphState):
    global memory_manager, r_memory_manager
    
    messages = state["messages"]
    code = state["code_solution"]
    lang = state["lang"] # Default to Python if language not specified
    
    stdout_capture = io.StringIO()
    try:

            
        if lang.lower() == "python":
            with warnings.catch_warnings(record=True) as captured_warnings, redirect_stdout(stdout_capture):
                warnings.simplefilter("default")  # Capture all warnings
                # Execute Python code as before
                local_namespace = memory_manager.get_namespace()
                exec(code, {}, local_namespace)
                memory_manager.update(local_namespace)
                stdout_output = stdout_capture.getvalue()[:2000] 
        elif lang.lower() == "r":
            if r_memory_manager is None:
                r_memory_manager = REnvironmentManager()

            py_namespace = memory_manager.get_namespace()
            result, stdout_output = r_memory_manager.execute_r_code_with_output(code, py_namespace)
            updated_namespace = r_memory_manager.update_to_python(py_namespace)
            memory_manager.update(updated_namespace)
            
            if result is not None:
                py_namespace['r_result'] = result
                memory_manager.update(py_namespace)
            if stdout_output:
                stdout_output = "\n".join(stdout_output)[:2000] 

        else:
            raise ValueError(f"Unsupported language: {lang}. Supported languages are 'python' and 'r'.")
                
    except (Exception, BaseException) as e:
        print(f"---Code checking ({lang}): failed---")
        error_message = traceback.format_exc()
        warning_message = ""
        if 'captured_warnings' in locals() and captured_warnings:
            warning_message = "The warning messages:\n" + "\n".join([str(warning.message) for warning in captured_warnings])
        if len(error_message)>10000:
            error_message=error_message[:5000]+error_message[-5000:]
        error_content = f"Your solution failed the {lang} code execution test:\n{error_message}\n{warning_message[:1500]}\n"

        # error_message = [("user", error_content)]
        error_message = HumanMessage(content= error_content)
        # messages.append(error_message) 
        
        if logger:
            logger.log_agent_output(error_content, f"code_execution_error", "user")
        
        return {
            "messages": [error_message],
            "error": 1,
        }


    if stdout_output:
        if logger:
            logger.log_agent_output(stdout_output, f"code_execution_success", "user")
        
        return {
            "messages": [stdout_output],
            "error": 0,            
        }
    else:
        if logger:
            logger.log_agent_output("", f"code_execution_success", "user")
        return {
            "error": 0,
        } 


def planner(state: GraphState):    
    current_step = state["current_step"]    
    if current_step == 0:
        print(f"---Generate plan---")
        plan = plan_chain.invoke({"input": [("user", state["input"])]})
        # syntax = Syntax(json.dumps(plan.steps, indent=4), "python", theme="monokai", line_numbers=True, word_wrap=True)
        # console.print(syntax)
        
        if logger:
            logger.log_agent_output(plan.steps, "planner", "assistant")
            logger.log_agent_output(plan.steps[current_step], "planner_steps", "assistant")
        
        print("---Current subplan {}---\n{}".format(current_step+1, plan.steps[current_step]))
        return {"plan": plan.steps, "current_plan": plan.steps[0],"current_step": current_step+1, "iterations":0,"error":0}
    else:
        if logger:
            logger.log_agent_output(state["plan"][current_step], "planner_steps", "assistant")

        print("---Current subplan {}---\n{}".format(current_step+1, state["plan"][current_step]))       
        return {"current_plan": state["plan"][current_step], "current_step": current_step+1, "iterations":0, "error":0}
    

@tool
def retrieve_tool(query: str) -> str:   
    """Retrieve the most relevant bioinformatics package name based on the input query."""
    result = vector_store.similarity_search_with_relevance_scores(query, k=1)
    result = [r for r in result if r[1]>0.2]
    
    print("---------call to retriever_tool---------")
    print(f"retrieve query: {query}")
    for res in result:
        console.print(f"retrieve result: {res[1]} [{res[0].metadata}] \n{res[0].page_content[:150]}")
    print("---------call to retriever_tool end---------")
    if logger:
        if result:
            res = result[0]
            logger.log_agent_output(f"{{'query':{query}, 'score':{res[1]}, 'retrieved_content':{res[0].page_content}}}", "rag", "tool")
        else:
            logger.log_agent_output(f"{{'query':{query}}}", "rag", "tool")

    return result

tools = [retrieve_tool]

def call_tools(state: GraphState):   
    print("---call tools---")    
    model = llm.bind_tools(tools)
    # response = model.invoke(state["current_plan"])
    response = ToolCaller(model)._create_chain().invoke({"current_plan":state["current_plan"]})
    console.print(response)
    
    return {"messages": [response]}