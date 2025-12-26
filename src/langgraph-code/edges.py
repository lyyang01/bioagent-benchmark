from graph import GraphState

max_iterations=5
reflect_flag=0


def tools_condition_judge(state: GraphState):
    if hasattr(state["messages"][-1], "tool_calls") and len(state["messages"][-1].tool_calls) > 0:
        return "tools"   
    else:
        return "coder"


def decide_to_finish(state: GraphState):
    iterations = state["iterations"]
    error = state["error"]
    total_steps = len(state["plan"])
    current_step = state["current_step"]
    if current_step < total_steps:
        if error == 0 and iterations <= max_iterations+1:
            print("---Decision: finish step {}---".format(current_step))
            return "planner"
        elif iterations <= max_iterations:
            print("---Decision: retry step {} for {} times---".format(current_step, iterations))
            return "coder"
        else:
            print("---Decision: max attempts exceeded at step {}---".format(current_step))
            return "end"
    else:
        if error == 1 and iterations <= max_iterations:
            print("---Decision: retry step {} for {} times---".format(current_step, iterations))
            return "coder"  
        elif error ==1:
            print("---Decision: max attempts exceeded at step {}---".format(current_step))
            return "end"
        else:         
            print("---Decision: finish all---")
            return "end"
    
