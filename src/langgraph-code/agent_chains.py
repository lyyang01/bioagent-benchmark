from typing import List
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import BaseModel, Field


class CodeOutput(BaseModel):
    code: str = Field(description="Code block including import statements")


class CodeGenerator:
    """Handles code generation with retries and error handling"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def _create_chain(self):
        """Initialize all code generation chains"""

        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coder with bioinformatics expertise tasked with generating fully executable Python/R code based on the provided messages. These messages contain the user’s original question, critical parameters, relevant datasets, API documentation, error messages, and other essential details. Carefully review and explicitly integrate this information into your generated code. If beneficial, utilize GPU configurations to accelerate training speed.

                    Important Note: The code necessary for this task might have already been implemented in previous steps. Carefully review previous code implementations provided in the messages, and reuse existing variables, functions, imported packages, or defined methods whenever possible. Do not regenerate or duplicate code already completed in earlier steps.

                    Your generated code must meet the following criteria:
                        1.	Directly address any remaining requirements of the current code-generation task that have not yet been implemented, based on previous steps and messages.
                        2.	Explicitly incorporate and correctly utilize important parameters, datasets, API details, or Python/R package documentation provided in the messages.
                        3.	Reuse previously defined variables, functions, imports, or code blocks whenever possible, avoiding redundant or duplicate code.
                        4.	Be logically structured, clearly defined, and include only new imports, definitions, or implementation steps that are necessary.
                        5.	Be directly executable without errors, ensuring all dependencies, variables, and functions are correctly referenced. Include all necessary imports and variable definitions.
                        6. If you want to view variable contents in python, use print(), e.g., 'print(adata)'. In R, typing a variable name alone will display its content.
    
                    **Please only return raw, executable codeas as plain text, without any formatting such as Markdown, JSON,  or triple backticks—only the plain code itself.**
                    The specific code-generation task:
                    {current_plan}

                    The provided messages:
                    {messages}
                    """,
                ),
                # ("placeholder", "{messages}"),
            ]
        )           

        # Build Chain with output check
        return self.code_gen_prompt | self.llm#.with_structured_output(CodeOutput)


class PlanOutput(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class PlanGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def _create_chain(self):
        self.planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a planner with bioinformatics expertise and need to design a concise analysis pipeline based on the provided experimental goals, data characteristics, and additional details from the messages. When making plans:

                        1. Focus strictly on essential computational steps required to achieve the core objectives.
                        2. Prioritize industry-standard tools with minimal redundancy.
                        3. Exclude quality control/validation steps unless explicitly requested.
                        4. Avoid tool comparisons, parameter optimizations, or exploratory analyses unless critical to the primary goal.
                        5. Present steps as a streamlined workflow using minimal sequential phases.
                        6. Clearly state the specific tool (e.g., Python/R package or software) required to implement each step.
                        7. If provided messages contain information about data file paths or parameters, integrate them clearly into the relevant steps of your plan.
                        8. Start each step of your plan with “Step …”.
                        9. Do not include any code in your planning.

                        Example of a good response integrating provided data paths or parameters(RNA-seq differential expression analysis):

                        Step 1. Raw read processing of input FASTQ files from `/data/sample_R1.fastq.gz` and `/data/sample_R2.fastq.gz` using Fastp (Fastp software), trimming adapters specified in provided parameters.
                        Step 2. Alignment to the provided human genome reference (`/refs/hg38.fa`) using STAR (STAR software), applying splice-junction parameters from provided information.
                        Step 3. Gene quantification with featureCounts (Subread package), utilizing the GTF annotation file specified at `/refs/hg38.gtf`.
                        Step 4. Differential expression analysis using DESeq2 (R package DESeq2), incorporating provided sample-group assignments and significance thresholds.

                        Keep each step justification to one sentence. Omit version specifications, intermediate file conversions, and redundant output formats unless explicitly required. Ensure the plan is concise, clear, and directly addresses the user query and messages.

                        Your response must be a valid JSON object with the following format:
                        {{
                        "steps": ["step 1", "step 2", "step 3", ...]
                        }}
                        Ensure you return ONLY the JSON with no additional text or explanation.
                    """,
                ),
                ("placeholder", "{input}"),
            ]
        )
        return self.planner_prompt | self.llm.with_structured_output(PlanOutput, method="json_mode")



class ToolCaller:
    def __init__(self, llm):
        self.llm = llm
    
    def _create_chain(self):
        self.planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an intelligent agent designed to trigger a retrieval tool for bioinformatics package when necessary and to generate accurate answers based on retrieved information.
                    current_plan: {current_plan}
                    """,

                )
            ]
        )
        return self.planner_prompt | self.llm    

