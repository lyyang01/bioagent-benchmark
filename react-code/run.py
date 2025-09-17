# from langchain.agents import AgentExecutor, create_react_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from logger_manager import init_logger  # Modified import
from utils import serialize_to_json, kill_child_processes
from config import initialize_llm
from typing import Annotated
from typing_extensions import TypedDict
import argparse, json, traceback
import os, sys
import asyncio


def create_react(llm):
    from tools import code_exec, code_generator, planning, code_retrieve
    prompt_template = ChatPromptTemplate.from_messages([

        ("system", '''
            You are a bioinformatics expert with extensive experience in planning, retrieving, generating, executing, and debugging code for bioinformatics tasks.

            Your workflow follows these steps:

            1. Plan Generation
            Develop a concise plan based on the given task, breaking it into steps. Focus strictly on essential computational steps required to achieve the core objectives. Prioritize industry-standard tools with minimal redundancy. For each step, specify the necessary tool name (e.g., Python/R package or software). Present a streamlined workflow in sequential phases starting with "Plan for..." and using "Step..." for each step. Exclude quality control/validation steps, tool comparisons, parameter optimizations, or exploratory analyses unless explicitly requested or critical to the primary goal. If provided messages contain information about data file paths or parameters, integrate them clearly into the relevant steps. Keep each step justification to one sentence. Ensure the plan contains no code.

            2. Step-by-Step Execution
            For each step in the plan:
            a) First determine if you need to retrieve existing bioinformatics tool API code examples. If the current step involves standard bioinformatics tools (such as scVI, Leiden, scanVI, etc.), use the code_retrieve tool with the appropriate tool name as query to find relevant code snippets and best practices. 
            b) If code_retrieve tool fails to return relevant code for a step, directly generate the necessary code without repeated retrieval attempts. Limit code_retrieve usage to one call per unique tool or library.
            c) Generate and display the comprehensive code tailored to the step's requirements, incorporating:
                - Retrieved tool API examples if available and relevant (filtered for accuracy and conciseness)
                - Parameters and datasets specified in the messages
                - Any previously defined variables, functions, imports, or code blocks to avoid redundancy
                - All necessary imports and variable definitions for direct execution
            d) Execute the code to validate its functionality using the code_exec tool.
            e) If any errors occur during execution, systematically debug and refine the code until the step is successfully completed.
            f) Proceed to the next step only after successfully completing the current step.

            IMPORTANT: After generating the plan, you MUST immediately begin executing the first step without waiting for further instructions. Do not stop after plan generation. Continue executing each step one by one until all steps are complete. The task is NOT complete until you have executed working code for every step in your plan.

            Continue this process until all steps in the plan have been successfully executed through working code. The task will only be considered complete when every step has been implemented with functional code that executes without errors and produces the expected results.  

            Task: {input}
            History messages: {messages}
        ''')])


    class CustomState(TypedDict):
        input: str
        messages: Annotated[list[BaseMessage], add_messages]
        is_last_step: IsLastStep
        remaining_steps: RemainingSteps

    # Construct the ReAct agent
    tools=[code_exec, code_retrieve]
    agent = create_react_agent(llm, tools, state_schema=CustomState, prompt=prompt_template)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=20, verbose=True, handle_parsing_errors=True)
    return agent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use")
    parser.add_argument("--task", type=str, default="gimvi", help="Workflow to run")
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--lab", type=str)
    parser.add_argument("--fw", type=str, default="react")
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
    llm = initialize_llm(model=args.model)

    # Log input data
    logger.log_agent_output(input_data, "main", "user")
    print(">>>>>>>>> Parameters <<<<<<<<<<<")
    print(f"Lab: {args.lab}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Lang: {input_lang}")
    print(f"Env: {input_env}")
    print(f"Framework: {args.fw}")

    try:
        timeout_s = 3*60*60


        async def run_graph():
            async for s in  create_react(llm).astream({"input": input_data, "lang": input_lang}, {"recursion_limit": 50}):
                logger.log_agent_output(serialize_to_json(s), "react_agent", "assistant")
        
        task = asyncio.create_task(run_graph())
        await asyncio.wait_for(task, timeout=timeout_s)  


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
        # Wait for the task to actually terminate
        try:
            await task
        except asyncio.CancelledError:
            pass  # This is expected when we cancel the task
        sys.exit(1)
    except Exception as e:
        if 'task' in locals():
            task.cancel()  # Also cancel the task for other exceptions
        error_msg = f"Error during workflow execution: {traceback.format_exc()}"
        logger.log_agent_output(error_msg, "error_msg", "system")
        sys.exit(1)



if __name__ == "__main__":
    # main()
    asyncio.run(main())