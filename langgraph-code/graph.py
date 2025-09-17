import operator
from typing import List, Annotated, List, Tuple, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        iterations : Number of tries
    """
    input: str
    plan: List[str]
    # past_steps: Annotated[List[Tuple], operator.add]
    current_step: int
    current_plan: str
    # context: str

    error: int
    # messages: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code_solution: str
    iterations: int
    shared_namespace: dict
    lang: str
    env: str


