import json
import argparse
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from edges import decide_to_finish
import traceback
from model_config import initialize_llm
from logger_manager import init_logger  
from utils import kill_child_processes
import asyncio
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
def create_workflow():

    from nodes import GraphState, planner, call_tools, code_generate, code_execution, retrieve_tool

    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("retrieve_tool", ToolNode([retrieve_tool]))
    workflow.add_node("call_tools", call_tools)
    workflow.add_node("coder", code_generate)
    workflow.add_node("code_execution", code_execution)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "call_tools")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "call_tools",
        tools_condition,
        {"tools": "retrieve_tool", END: "coder"}
    )
    
    workflow.add_edge("retrieve_tool", "coder")
    workflow.add_edge("coder", "code_execution")
    
    workflow.add_conditional_edges(
        "code_execution",
        decide_to_finish,
        {"end": END, "coder": "coder", "planner": "planner"}
    )
    
    return workflow.compile(checkpointer=MemorySaver())

async def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--llm_provider", type=str, default="azure", help="LLM provider to use")
    parser.add_argument("--model", type=str, help="Model name to use") #, default="gpt-4o"
    parser.add_argument("--task", type=str, help="Workflow to run") #, default="scanvi" # singlecellhaystack
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--lab", type=str)
    parser.add_argument("--fw", type=str, default="langgraph")
    args = parser.parse_args()
    
    # Initialize logger using init_logger
    logger = init_logger(model=args.model, task=args.task, lab=args.lab)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Create model and load input
    try:
        input_json = json.load(open("../prompt_gradient.json", "r")).get(args.task)
        if args.lab  == "gradient_prompt/intermediate":
            input_data = input_json["prompt_input"]["intermediate"]
        elif args.lab == "gradient_prompt/advanced":
            input_data = input_json["prompt_input"]["advanced"]
        else:
            input_data = input_json["prompt_input"]["basic"]
        input_env = input_json["conda_env"]
        input_lang = input_json["language"]
        input_data += f"\nThe code languange is {input_lang}.\nAll output files must save in the path: ../results/{str(args.lab)}/{str(args.fw)}/{str(args.model)}/{str(args.task)}/agent_output"
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        error_msg = f"Error loading input data: {str(e)}"
        logger.log_agent_output(error_msg, "error_msg", "system")
        print(error_msg)
        return None
    
    # Initialize LLM
    initialize_llm(model=args.model)
    
    # Log input data
    logger.log_agent_output(input_data, "main", "user")
    print(">>>>>>>>> Parameters <<<<<<<<<<<")
    print(f"Lab: {args.lab}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Lang: {input_lang}")
    print(f"Env: {input_env}")
    print(f"Input: {input_data[:100]}...")
    
    # Run graph
    graph_config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
    
    try:
        timeout_s = 3*60*60

        async def run_graph():
            async for chunk in create_workflow().astream(
                {"input": input_data, "current_step": 0, "lang": input_lang},
                graph_config,
                stream_mode="values"                
            ):
                pass   

        task = asyncio.create_task(
            run_graph()
        )
        await asyncio.wait_for(task, timeout=timeout_s)      

        # Log results
        logger.log_result({})
        print("\n--- Final Result ---")

        
        # Save formatted logs
        formatted_logs = logger.get_formatted_logs()
        print(f"\nLogs saved to: {logger.log_path}")
        sys.exit(0)

    except asyncio.TimeoutError:
        task.cancel()
        error_msg = f"Time Out {timeout_s} s.Task Cancelled."
        logger.log_agent_output(error_msg, "error_msg", "system")
        kill_child_processes()
        try:
            await task
        except asyncio.CancelledError:
            pass  # This is expected when we cancel the task
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error during workflow execution: {traceback.format_exc()}"
        logger.log_agent_output(error_msg, "error_msg", "system")
        sys.exit(1)



if __name__ == "__main__":
    # main()
    asyncio.run(main())