# 🧩 Error Log Files

This folder contains JSON files representing different **error types** observed during multi-agent task execution.  
Each file documents anonymized examples or summaries for a specific error type.

---

## 📘 File Naming Convention

`<error_type>.<framework>_<model>_<task>.json`

**Examples:**

ambiguous_input_prompt.langgraph_grok3-beta_deeptree.json

Explanation:
- `<error_type>` — standardized error name in lowercase with underscores  
- `<framework>` — agent orchestration framework (e.g., `langgraph`, `react`, `autogen`)  
- `<model>` — language model used (e.g., `gpt-4o`, `deepseek-v3`, `sonnet-3.7`)  
- `<task>` — specific task or dataset identifier (e.g., `decoupler`, `seurat-1`, `cell2location`)

---


## 📂 Error Types

| **Error Type** | **Description** |
|----------------|-----------------|
| **Interrupted Execution** | Network failures, API limits, or program crashes. |
| **Exceeded Round/Time Limit** | Excessive execution time, repeated planning, unfinished tasks, or reaching maximum round limit. |
| **Knowledge Acquisition Failure** | Failure to invoke correct APIs, generate valid code, or access essential task info. |
| **Long Context Handling Failure** | Ignoring earlier content, context loss, erratic behavior, or internal inconsistencies. |
| **Information Not Shared or Unsynced** | Missing or unsynchronized information between agents (e.g., inputs, outputs, dependencies). |
| **Inconsistent Planning Behavior** | Divergence between planner and executor, or missing critical procedural steps. |
| **Lack of Active Clarification** | No clarification requests for key unknowns (e.g., variable types, structures, print steps). |
| **Early Stopping Fault** | Task terminated prematurely before executing key steps or verifying completion. |
| **Ambiguous Input Prompt** | Vague or underspecified task instructions leading to unstable model outputs. |
| **Missing Environment Dependency** | Missing packages, failed imports, or incorrect tool paths. |
| **Planner Issue** | Incomplete or faulty task planning, missing preprocessing steps, or poor process coverage. |
| **Coder Issue** | Incorrect parameters, unsaved outputs, or flawed code logic. |
| **Poor Instruction Following** | Failing to follow task instructions (e.g., wrong save path, missing boundary handling). |
| **Redundant Process or Code** | Repeated code generation, redundant module calls, or failure to skip ineffective steps. |


