import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psutil
import subprocess
import time
import numpy as np

import os
import openai
import json
from evaluation.model_config_eval import CONFIG_LIST




def get_gpu_usage(gpu_id=0):
    """通过 nvidia-smi 查询整个系统的 GPU 使用情况和显存使用量"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpu_usages = result.stdout.strip().split("\n")
    gpu_util = []
    gpu_memory_used = []
    #import pdb
    #pdb.set_trace()
    for usage in gpu_usages:
        util, memory_used = usage.split(", ")
        gpu_util.append(float(util))
        gpu_memory_used.append(int(memory_used))  # 单位为 MiB
    #import pdb
    #pdb.set_trace()
    #import pdb
    return gpu_util[gpu_id], gpu_memory_used[gpu_id]

##---------------------Monitor-----------------------------------  
def monitor_process(script_path, tool_name, model_name, result_type, prompt_path, result_path, log_file):
    """启动脚本并监控其资源使用情况，将结果保存到日志文件"""
    #process = subprocess.Popen(["python", script_path, "--task", tool_name])
    process = subprocess.Popen(["python", script_path, "--task", tool_name, "--model", model_name, "--lab", result_type, "--prompt_path", prompt_path, "--result_path", result_path])
    
    target_pid = process.pid
    print(f"Target script PID: {target_pid}")

    # 初始化监控数据
    start_time = time.time()
    max_cpu_usage = 0
    max_memory_usage_percent = 0
    max_memory_usage_kb = 0
    max_gpu_util = 0
    max_gpu_memory = 0

    # 初始化 CPU 时间
    user_time = 0.0
    system_time = 0.0
    #log_file = os.path.join(log_path, f"{tool_name}_monitor_log.csv")
    # 打开日志文件
    with open(log_file, "w") as log:
        log.write("Timestamp, CPU Usage (%), Memory Usage (%), Memory Usage (KB), GPU Utilization (%), GPU Memory (MB)\n")

        try:
            while process.poll() is None:
                # 获取目标进程的 CPU 和内存使用率
                cpu_usage = psutil.Process(target_pid).cpu_percent(interval=1)
                memory_usage_percent = psutil.Process(target_pid).memory_percent()
                memory_usage_kb = psutil.Process(target_pid).memory_info().rss / 1024  # 转换为 KB
                
                # 获取 GPU 使用情况
                gpu_util, gpu_memory = get_gpu_usage()
                #gpu_util = get_gpu_usage()
                if gpu_util is not None:
                    max_gpu_util = max(max_gpu_util, gpu_util)
                    max_gpu_memory = max(max_gpu_memory, gpu_memory)

                # 更新最大值
                max_cpu_usage = max(max_cpu_usage, cpu_usage)
                max_memory_usage_percent = max(max_memory_usage_percent, memory_usage_percent)
                max_memory_usage_kb = max(max_memory_usage_kb, memory_usage_kb)

                # 获取 CPU 时间（用户态和内核态）
                cpu_times = psutil.Process(target_pid).cpu_times()
                #cpu_times = psutil.cpu_times()
                user_time = cpu_times.user
                system_time = cpu_times.system

                # 记录当前时间戳和监控数据
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                time.sleep(1)
        finally:
            # 计算运行时间
            elapsed_time = time.time() - start_time

            # 计算 CPU Usage
            cpu_usage_percentage = ((user_time + system_time) / elapsed_time) * 100

            # 记录最终结果
            log.write(f"\nScript finished. Total runtime: {elapsed_time:.2f} seconds\n")
            log.write(f"User Time: {user_time:.2f} seconds\n")
            log.write(f"System Time: {system_time:.2f} seconds\n")
            log.write(f"CPU Usage: {cpu_usage_percentage:.2f}%\n")
            log.write(f"Max CPU Usage: {max_cpu_usage:.2f}%\n")
            log.write(f"Max Memory Usage (%): {max_memory_usage_percent:.2f}%\n")
            log.write(f"Max Memory Usage (KB): {max_memory_usage_kb:.2f} KB\n")
            log.write(f"Max GPU Util: {max_gpu_util:.2f}%\n")
            log.write(f"Max GPU Memory: {max_gpu_memory} MB\n")

            print(f"Script finished. Total runtime: {elapsed_time:.2f} seconds")
            print(f"User Time: {user_time:.2f} seconds")
            print(f"System Time: {system_time:.2f} seconds")
            print(f"CPU Usage: {cpu_usage_percentage:.2f}%")
            print(f"Max CPU Usage: {max_cpu_usage:.2f}%")
            print(f"Max Memory Usage (%): {max_memory_usage_percent:.2f}%")
            print(f"Max Memory Usage (KB): {max_memory_usage_kb:.2f} KB")
            print(f"Max GPU Util: {max_gpu_util:.2f}%")
            print(f"Max GPU Memory: {max_gpu_memory} MB")
##---------------------Monitor-----------------------------------

##---------------------Plan Scores-----------------------------------
def plan_score(plan_str, tool_name, prompt_path, eval_model="gpt-4o"):
    config_list = CONFIG_LIST[eval_model]
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['base_url']

    openai.api_key = config_list[0]['api_key']
    
    #read prompt.json
    import json
    with open(prompt_path) as f:
        user_prompt = json.load(f)
    #with open(plan_file_path) as f:
    #    plan_str = f.read()

    task = user_prompt[tool_name]["prompt_input"]

    prompt = '''
You are tasked with evaluating a bioinformatics task plan based on the task description, the plan content and predefined attributes. Please follow the guidelines below to provide a structured evaluation:
1. Evaluate the provided bioinformatics task plan. The plan should ideally include the following components:
(1) Data Preprocessing: Steps for preparing the data.
(2) Model Building and Training: Description of the model architecture, training process, and any hyperparameter tuning.
(3) Model Evaluation: Metrics and methods used to assess the model's performance.
(4) Post-analysis: Any additional analysis or interpretation of results.
Content Evaluation Criteria:
Rate each content component on a scale of 1 to 5, where:
    1~2 = Poor (Missing or severely inadequate)
    2~3 = Fair (Incomplete or insufficient details)
    3~4 = Good (Adequate but could be improved)
    4~5 = Very Good (Comprehensive with minor improvements needed)
    5 = Excellent (Complete and well-detailed)
2. Attributes to Evaluate:
(1) Clarify (Plan Clarity): Is the plan easy to understand? Are the steps clearly described?
(2) Comprehensiveness (Plan Completeness): Does the plan cover all necessary aspects of the task?
(3) Structurality (Plan Structure): Is the plan well-organized? Does it follow a logical sequence?
(4) Details (Plan Details): Are the steps described in sufficient detail? Are there specific methods or tools mentioned?
(5) Technical Feasibility (Technical Feasibility): Is the plan realistic given current technology and resources?
Attributes Evaluation Criteria:
Rate each attribute on a scale of 1 to 5, where:
    1~2 = Poor
    2~3 = Fair
    3~4 = Good
    4~5 = Very Good
    5 = Excellent

Output Format:
Provide your evaluation in the following structured format:

Evaluation of Bioinformatics Task Plan (strictly in json format):
```json
{
    "Content":
    {
        "Data_Preprocessing": [Score],
        "Model_Building_and_Training": [Score],
        "Model_Evaluation": [Score],
        "Post-analysis": [Score],
    },
    "Attributes":
    {
        "Clarify": [Score],
        "Comprehensiveness": [Score],
        "Structurality": [Score],
        "Details": [Score],
        "Technical_Feasibility": [Score],
    }
}
```json
Summary:
   - Briefly summarize your overall assessment of the plan.

Now, I provide you with the bioinformatics task description and the plan for scoring.
The bioinformatics task description:\n'''+f"{task}\n"+"The plan by agent:\n"+f"{plan_str}"

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    #import pdb
    #pdb.set_trace()

    tmp = []
    json_str = ""
    indic = 0

    #import pdb
    #pdb.set_trace()
    #print(model_reply)
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        import re
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        indic += 1
    #import pdb
    #pdb.set_trace()
    if type(json_str) == dict:
        tmp_ = json_str
    else:
        tmp_ = eval(json_str)
    content_score = tmp_["Content"]
    attribute_score = tmp_["Attributes"]
    
    plan_scores = {"content":{}, "attribute":{}}
    score_tmp = 0
    for key_, score_ in content_score.items():
        score_tmp += score_
            

    for key_, score_ in attribute_score.items():
        score_tmp += score_
    plan_scores["content"] = content_score
    plan_scores["attribute"] = attribute_score

    avg_overall = float(score_tmp) / float(9)
    plan_scores["avg_overall"] = avg_overall

    return plan_scores
    #with open(os.path.join(output_path, f"{tool_name}_plan_score.json"), "w", encoding="utf-8") as f:
    #    json.dump(plan_score, f)
##---------------------Plan Scores-----------------------------------

##---------------------Code Scores-----------------------------------
def code_score(code_file_path, tool_name, prompt_path, gt_path, eval_model="gpt-4o"):
    config_list = CONFIG_LIST[eval_model]
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']

    openai.api_key = config_list[0]['api_key']
    import json
    #read prompt.json
    with open(prompt_path) as f:
        user_prompt = json.load(f)
    
    task = user_prompt[tool_name]["prompt_input"]
    
    with open(code_file_path) as f:
        code_str = f.read()
    #if code_str == "":
        
    if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
        with open(f"{gt_path}/{tool_name}_gold.r") as f:
            gold_code_str = f.read()
    else:
        with open(f"{gt_path}/{tool_name}_gold.py") as f:
            gold_code_str = f.read()

    clean_code_str = ""
    clean_gold_code_str = ""
    for i in code_str.split("\n"):
        if i.startswith("#"):
            continue
        if i == "":
            continue
        clean_code_str += i + "\n"

    for i in gold_code_str.split("\n"):
        if i.startswith("#"):
            continue
        if i == "":
            continue
        clean_gold_code_str += i + "\n"

    code_scores = {"gpt_score":{}, "obj_score":{}}
    prompt = '''
You are tasked with evaluating a bioinformatics task code by comparing it with a reference code. Please follow the guidelines below to provide a structured evaluation.
Task:
Evaluate the provided bioinformatics task code based on the following attributes:
(1) Key Segment Matching Degree: How well does the code match the key segments of the reference code? Are essential parts of the reference code present and correctly implemented?
(2) Efficiency: Is the code optimized for performance? Does it use efficient algorithms and data structures?
(3) Completeness: Does the code cover all necessary functionalities and tasks as required? Are there any missing parts?
(4) Readability: Is the code easy to read and understand? Are there proper comments, variable names, and formatting?
(5) Logicality: Is the code logically structured? Are the steps and processes well-organized and coherent?
Evaluation Criteria:
Rate each attribute on a scale of 1 to 5, where:
    1~2 = Poor
    2~3 = Fair
    3~4 = Good
    4~5 = Very Good
    5 = Excellent

Output Format:
Provide your evaluation in the following structured format:

1. Evaluation of Bioinformatics Task Code (strictly in json format):
```json
{
   "Key_Segment_Matching_Degree": [Score],
   "Efficiency": [Score],
   "Completeness": [Score],
   "Readability": [Score],
   "Logicality": [Score],
}
```json
2. Summary:
   - Briefly summarize your overall assessment of the code.

Now, I provide you with the bioinformatics task description, the reference code and the code written by agent for scoring.
The bioinformatics task description:\n'''+f"{task}\n" + "The reference code:\n"+ f"{gold_code_str}\n" + "The code by agent:" + f"{code_str}"
    if clean_code_str == "":
        code_scores["gpt_score"]["Key_Segment_Matching_Degree"] = 0.0
        code_scores["gpt_score"]["Efficiency"] = 0.0
        code_scores["gpt_score"]["Completeness"] = 0.0
        code_scores["gpt_score"]["Readability"] = 0.0
        code_scores["gpt_score"]["Logicality"] = 0.0
    else:
        #response = openai.ChatCompletion.create(
        response = openai.chat.completions.create(
        model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
        messages = [{"role":"user", "content": prompt}],
        temperature=0,
        #max_tokens=2048,
        )
        indic = 0
        json_str = ""
        while(json_str==""):
            if indic > 3:
                break
            model_reply = response.choices[0].message.content
            #print(model_reply)
            import re
            tmp = model_reply.split("\n\n")[0]
            tmp = "".join(tmp.split("\n")[1:])
            tmp = "".join(tmp.split(" "))
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, tmp, re.DOTALL)
            json_str = '\n'.join(matches)
            if json_str == "":
                try:
                    json_str = eval(model_reply)
                except:
                    pass
            indic += 1
        #if json_str == "":
        #    json_str = tmp

        #import pdb
        #pdb.set_trace()
        if type(json_str) == dict:
            gpt_scores = json_str
        else:
            try:
                gpt_scores = eval(json_str)
            except:
                gpt_scores = {}
                gpt_scores["Key_Segment_Matching_Degree"] = 0.0
                gpt_scores["Efficiency"] = 0.0
                gpt_scores["Completeness"] = 0.0
                gpt_scores["Readability"] = 0.0
                gpt_scores["Logicality"] = 0.0
        
        for key_, score_ in gpt_scores.items():
            code_scores["gpt_score"][key_] = score_


    
    ##similarity detection
    #import pycode_similar
    #rouge
    from rouge import Rouge
    rouge = Rouge()
    if clean_code_str == "":
        code_scores["obj_score"]["rouge"] = [0.0, 0.0, 0.0]
    else:
        scores = rouge.get_scores(clean_code_str, clean_gold_code_str)
    #print("ROUGE-1:", scores[0]['rouge-1'])
    #print("ROUGE-2:", scores[0]['rouge-2'])
    #print("ROUGE-L:", scores[0]['rouge-l'])

        code_scores["obj_score"]["rouge"] = [scores[0]['rouge-1'], scores[0]['rouge-2'], scores[0]['rouge-l']]

    #ast similarity
    import ast
    import difflib

    def parse_code_to_ast(code):
        """
        将代码字符串解析为抽象语法树（AST）。
        
        :param code: 代码字符串
        :return: AST 对象
        """
        try:
            tree = ast.parse(code)
            return tree
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return None

    def ast_to_string(tree):
        """
        将 AST 对象转换为字符串表示，便于比较。
        
        :param tree: AST 对象
        :return: AST 的字符串表示
        """
        def _traverse(node):
            if isinstance(node, ast.AST):
                return f"{type(node).__name__}({', '.join(_traverse(child) for child in ast.iter_child_nodes(node))})"
            return repr(node)
        
        return _traverse(tree)

    def calculate_ast_similarity(reference_code, generated_code):
        """
        使用 AST 比较两段代码的相似度。
        
        :param reference_code: 参考代码字符串
        :param generated_code: 生成代码字符串
        :return: 相似度分数（0到1之间）
        """
        reference_ast = parse_code_to_ast(reference_code)
        generated_ast = parse_code_to_ast(generated_code)
        
        if reference_ast is None or generated_ast is None:
            return 0.0
        
        reference_ast_str = ast_to_string(reference_ast)
        generated_ast_str = ast_to_string(generated_ast)
        
        # 使用 difflib 计算相似度
        similarity = difflib.SequenceMatcher(None, reference_ast_str, generated_ast_str).ratio()
        return similarity

    similarity = calculate_ast_similarity(clean_gold_code_str, clean_code_str)
    #print(f"AST Similarity: {similarity:.2f} (1.0 means identical, 0.0 means completely different)")

    code_scores["obj_score"]["ast_sim"] = similarity

    return code_scores
    
##---------------------Code Scores-----------------------------------


##---------------------Cross Judge Code Scores-----------------------------------
def code_score_cross_judge(code_file_path, tool_name, prompt_path, gt_path, eval_model="gpt-4o"):
    config_list = CONFIG_LIST[eval_model]
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']

    openai.api_key = config_list[0]['api_key']
    import json
    #read prompt.json
    with open(prompt_path) as f:
        user_prompt = json.load(f)
    
    task = user_prompt[tool_name]["prompt_input"]
    
    with open(code_file_path) as f:
        code_str = f.read()
    #if code_str == "":
        
    if tool_name in ["cellchat", "seurat-1", "seurat-2", "sctransform", "singlecellhaystack", "scpnmf", "scry", "scorpius", "nichenet"]:
        with open(f"{gt_path}/{tool_name}_gold.r") as f:
            gold_code_str = f.read()
    else:
        with open(f"{gt_path}/{tool_name}_gold.py") as f:
            gold_code_str = f.read()

    clean_code_str = ""
    clean_gold_code_str = ""
    for i in code_str.split("\n"):
        if i.startswith("#"):
            continue
        if i == "":
            continue
        clean_code_str += i + "\n"

    for i in gold_code_str.split("\n"):
        if i.startswith("#"):
            continue
        if i == "":
            continue
        clean_gold_code_str += i + "\n"

    code_scores = {"gpt_score":{}, "obj_score":{}}
    prompt = '''
You are tasked with evaluating a bioinformatics task code by comparing it with a reference code. Please follow the guidelines below to provide a structured evaluation.
Task:
Evaluate the provided bioinformatics task code based on the following attributes:
(1) Key Segment Matching Degree: How well does the code match the key segments of the reference code? Are essential parts of the reference code present and correctly implemented?
(2) Efficiency: Is the code optimized for performance? Does it use efficient algorithms and data structures?
(3) Completeness: Does the code cover all necessary functionalities and tasks as required? Are there any missing parts?
(4) Readability: Is the code easy to read and understand? Are there proper comments, variable names, and formatting?
(5) Logicality: Is the code logically structured? Are the steps and processes well-organized and coherent?
Evaluation Criteria:
Rate each attribute on a scale of 1 to 5, where:
    1~2 = Poor
    2~3 = Fair
    3~4 = Good
    4~5 = Very Good
    5 = Excellent

Output Format:
Provide your evaluation in the following structured format:

1. Evaluation of Bioinformatics Task Code (strictly in json format):
```json
{
   "Key_Segment_Matching_Degree": [Score],
   "Efficiency": [Score],
   "Completeness": [Score],
   "Readability": [Score],
   "Logicality": [Score],
}
```json
2. Summary:
   - Briefly summarize your overall assessment of the code.

Now, I provide you with the bioinformatics task description, the reference code and the code written by agent for scoring.
The bioinformatics task description:\n'''+f"{task}\n" + "The reference code:\n"+ f"{gold_code_str}\n" + "The code by agent:" + f"{code_str}"
    if clean_code_str == "":
        code_scores["gpt_score"]["Key_Segment_Matching_Degree"] = 0.0
        code_scores["gpt_score"]["Efficiency"] = 0.0
        code_scores["gpt_score"]["Completeness"] = 0.0
        code_scores["gpt_score"]["Readability"] = 0.0
        code_scores["gpt_score"]["Logicality"] = 0.0
    else:
        #response = openai.ChatCompletion.create(
        response = openai.chat.completions.create(
        model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
        messages = [{"role":"user", "content": prompt}],
        temperature=0,
        #max_tokens=2048,
        )
        indic = 0
        json_str = ""
        while(json_str==""):
            if indic > 3:
                break
            model_reply = response.choices[0].message.content
            #print(model_reply)
            import re
            tmp = model_reply.split("\n\n")[0]
            tmp = "".join(tmp.split("\n")[1:])
            tmp = "".join(tmp.split(" "))
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, tmp, re.DOTALL)
            json_str = '\n'.join(matches)
            if json_str == "":
                try:
                    json_str = eval(model_reply)
                except:
                    pass
            indic += 1
        #if json_str == "":
        #    json_str = tmp

        #import pdb
        #pdb.set_trace()
        if type(json_str) == dict:
            gpt_scores = json_str
        else:
            try:
                gpt_scores = eval(json_str)
            except:
                gpt_scores = {}
                gpt_scores["Key_Segment_Matching_Degree"] = 0.0
                gpt_scores["Efficiency"] = 0.0
                gpt_scores["Completeness"] = 0.0
                gpt_scores["Readability"] = 0.0
                gpt_scores["Logicality"] = 0.0
        
        for key_, score_ in gpt_scores.items():
            code_scores["gpt_score"][key_] = score_
    return code_scores
##---------------------Cross Judge Code Scores-----------------------------------


##---------------------CPU -----------------------------------
def monitor_process_code(code_file_path, log_file, gpu_id):
    """启动脚本并监控其资源使用情况，将结果保存到日志文件"""
    #process = subprocess.Popen(["python", script_path, "--task", tool_name])
    # 设置环境变量以指定 GPU ID
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 使用修改后的环境变量启动进程
    process = subprocess.Popen(["python", code_file_path], env=my_env)

    # process = subprocess.Popen(["python", code_file_path])
    target_pid = process.pid
    print(f"Target script PID: {target_pid}")

    # 初始化监控数据
    start_time = time.time()
    max_cpu_usage = 0
    max_memory_usage_percent = 0
    max_memory_usage_kb = 0
    max_gpu_util = 0
    max_gpu_memory = 0

    # 初始化 CPU 时间
    user_time = 0.0
    system_time = 0.0

    # 打开日志文件
    with open(log_file, "w") as log:
        log.write("Timestamp, CPU Usage (%), Memory Usage (%), Memory Usage (KB), GPU Utilization (%), GPU Memory (MB)\n")

        try:
            while process.poll() is None:
                # 获取目标进程的 CPU 和内存使用率
                cpu_usage = psutil.Process(target_pid).cpu_percent(interval=1)
                memory_usage_percent = psutil.Process(target_pid).memory_percent()
                memory_usage_kb = psutil.Process(target_pid).memory_info().rss / 1024  # 转换为 KB
                #cpu_usage = psutil.cpu_percent(interval=1)
                #memory_usage_percent = psutil.memory_percent()
                #memory_usage_kb = psutil.memory_info().rss / 1024  # 转换为 KB
                #memory_info = psutil.virtual_memory()
                #memory_usage_percent = memory_info.percent
                #memory_usage_kb = memory_info.used / 1024  # 转换为 KB
                

                # 获取 GPU 使用情况
                #import pdb
                #pdb.set_trace()
                #gpu_util, gpu_memory = get_gpu_usage_by_pid(target_pid)
                gpu_util, gpu_memory = get_gpu_usage(gpu_id)
                #gpu_util = get_gpu_usage()
                if gpu_util is not None:
                    max_gpu_util = max(max_gpu_util, gpu_util)
                    max_gpu_memory = max(max_gpu_memory, gpu_memory)

                # 更新最大值
                max_cpu_usage = max(max_cpu_usage, cpu_usage)
                max_memory_usage_percent = max(max_memory_usage_percent, memory_usage_percent)
                max_memory_usage_kb = max(max_memory_usage_kb, memory_usage_kb)

                # 获取 CPU 时间（用户态和内核态）
                cpu_times = psutil.Process(target_pid).cpu_times()
                #cpu_times = psutil.cpu_times()
                user_time = cpu_times.user
                system_time = cpu_times.system

                # 记录当前时间戳和监控数据
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                #log.write(f"{timestamp}, {cpu_usage:.2f}, {memory_usage_percent:.2f}, {memory_usage_kb:.2f}, {max(gpu_usages) or 0:.2f}, {max(gpu_memory) or 0}\n")
                #log.write(f"{timestamp}, {cpu_usage:.2f}, {memory_usage_percent:.2f}, {memory_usage_kb:.2f}, {gpu_util or 0:.2f}, {gpu_memory or 0}\n")
                #log.flush()  # 确保数据实时写入文件
                
                #print(f"CPU: {cpu_usage:.2f}%, Memory: {memory_usage_percent:.2f}% ({memory_usage_kb:.2f} KB), "
                #      f"GPU Util: {gpu_util or 0:.2f}%, GPU Memory: {gpu_memory or 0} MB")
                #print(f"CPU: {cpu_usage:.2f}%, Memory: {memory_usage_percent:.2f}% ({memory_usage_kb:.2f} KB), "
                #      f"GPU Util: {max(gpu_usages) or 0:.2f}%, GPU Memory: {max(gpu_memory) or 0} MB")
                
                time.sleep(1)
        finally:
            # 计算运行时间
            elapsed_time = time.time() - start_time

            # 计算 CPU Usage
            cpu_usage_percentage = ((user_time + system_time) / elapsed_time) * 100

            # 记录最终结果
            log.write(f"\nScript finished. Total runtime: {elapsed_time:.2f} seconds\n")
            log.write(f"User Time: {user_time:.2f} seconds\n")
            log.write(f"System Time: {system_time:.2f} seconds\n")
            log.write(f"CPU Usage: {cpu_usage_percentage:.2f}%\n")
            log.write(f"Max CPU Usage: {max_cpu_usage:.2f}%\n")
            log.write(f"Max Memory Usage (%): {max_memory_usage_percent:.2f}%\n")
            log.write(f"Max Memory Usage (KB): {max_memory_usage_kb:.2f} KB\n")
            log.write(f"Max GPU Util: {max_gpu_util:.2f}%\n")
            log.write(f"Max GPU Memory: {max_gpu_memory} MB\n")

            print(f"Script finished. Total runtime: {elapsed_time:.2f} seconds")
            print(f"User Time: {user_time:.2f} seconds")
            print(f"System Time: {system_time:.2f} seconds")
            print(f"CPU Usage: {cpu_usage_percentage:.2f}%")
            print(f"Max CPU Usage: {max_cpu_usage:.2f}%")
            print(f"Max Memory Usage (%): {max_memory_usage_percent:.2f}%")
            print(f"Max Memory Usage (KB): {max_memory_usage_kb:.2f} KB")
            print(f"Max GPU Util: {max_gpu_util:.2f}%")
            print(f"Max GPU Memory: {max_gpu_memory} MB")
##---------------------CPU -----------------------------------


def task_completion_react(workflow_str, plan_str, root_path, tool_name, lang, eval_model="gpt-4o"):
    config_list = CONFIG_LIST[eval_model]

    import traceback
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    # Get structured steps from LLM

    prompt_plan = f'Extract the planning steps as: ```json ["Step 1 ...", "Step 2 ...", ...] ``` From: {plan_str}'
    # prompt_complete = f'List ONLY the completed steps as: ```json ["Step 1 ...", "Step 2 ..."] ```. From: {[i for i in workflow_str_list if i["name"] != "rag"]}'
    response = openai.chat.completions.create(
    model=config_list[0]['model'], 
    messages = [{"role":"user", "content": prompt_plan}],
    temperature=0)
    result = response.choices[0].message.content
    
    try:
        #print(result)
        plan_list = json.loads(result.split("```json")[1].split("```")[0])
    except Exception as e:
        #print(traceback.format_exc())
        raise ValueError("Failed to parse planning steps from LLM response") from e


    workflow_str_list = eval(workflow_str)
    completed_step = 0
    step_list = []
    import re
    for i,_ in enumerate(workflow_str_list):
        if workflow_str_list[i]["name"] == "code_execution_success":
            content_str = workflow_str_list[i]["content"]
            if "current_step" in content_str:
                pattern = r'"current_step"\s*:\s*([^,}]+)'
                match = re.search(pattern, content_str, re.DOTALL)
                if match:
                    current_step_value = match.group(1).strip()
                    # Optional: Remove quotes if they exist
                    if current_step_value.startswith('"') and current_step_value.endswith('"'):
                        current_step_value = current_step_value[1:-1].strip()
                    # Now extract the step number from the string
                    step_match = re.search(r"Step\s*(\d+)", current_step_value, flags=re.IGNORECASE)
                    if step_match:
                        step_number = int(step_match.group(1))
                        step_list.append(step_number)
                        # print("Extracted step number:", step_number)
                    else:
                        pass
                        # print("No step number found in current_step.")

    step_list=list(dict.fromkeys(step_list))   
    
    if not step_list:
        if lang=="R":
            code_file_path = os.path.join(root_path, f"{tool_name}_code.r")
        else:
            code_file_path = os.path.join(root_path, f"{tool_name}_code.py")
        with open(code_file_path, "r", encoding="utf-8") as f:
            code_content = f.read().strip()
        if code_content:
            prompt_code = (
                f'Given the plan: {plan_list}, identify which steps have already been completed based on the following code:\n\n'
                f'{code_content}\n\n'
                f'Return only the completed steps in this JSON format:\n'
                f'```json\n["Step 1 ...", "Step 2 ...", ...]\n```'
            )

            response = openai.chat.completions.create(
            model=config_list[0]['model'], 
            messages = [{"role":"user", "content": prompt_code}],
            temperature=0)
            result = response.choices[0].message.content
            
            try:
                step_list = json.loads(result.split("```json")[1].split("```")[0])
                #print(code_file_path)
                #print("plan_list")
                #print(plan_list)
                #print("completed_step")
                #print(step_list)
            except Exception as e:
                #print(traceback.format_exc())
                raise ValueError("Failed to parse completed steps from LLM response") from e

    completed_step=len(step_list)

    tc_dict = {}
    success_rate = 0
    for i in range(len(workflow_str_list)-1, -1 , -1):
        if workflow_str_list[i]["name"] == "error_msg" or workflow_str_list[i]["name"] == "erro_msg":
            success_rate = 0
            break            
        if workflow_str_list[i]["name"] == "code_execution_success":
            success_rate = 1
            break
        if workflow_str_list[i]["name"] == "code_execution_error":
            success_rate = 0
            break
    
    tc_dict["success_rate"] = success_rate
    tc_dict["all_step"] = plan_list
    all_step_num = len(plan_list)

    tc_dict["task_complete_rate"] = float(min(all_step_num, completed_step))/float(all_step_num)#task_complete
    tc_dict["step_number"] = all_step_num
    
    return tc_dict


def task_completion_react_noplan(workflow_str_list):
    tc_dict = {}
    success_rate = 0
    for i in range(len(workflow_str_list)-1, -1 , -1):
        if workflow_str_list[i]["name"] == "error_msg" or workflow_str_list[i]["name"] == "erro_msg":
            success_rate = 0
            break            
        if workflow_str_list[i]["name"] == "code_execution_success":
            success_rate = 1
            break
        if workflow_str_list[i]["name"] == "code_execution_error":
            success_rate = 0
            break
    
    tc_dict["success_rate"] = success_rate
    
    return tc_dict


def task_completion_langgraph(workflow_str):
    workflow_str_list = eval(workflow_str)
    completed_step = 0
    for i in workflow_str_list:
        if i["name"] == "planner":
            plan_list = i["content"]
        if i["name"] == "code_execution_success":
            completed_step += 1
    
    tc_dict = {}
    success_rate = 0
    for i in range(len(workflow_str_list)-1, -1 , -1):
        if workflow_str_list[i]["name"] == "error_msg" or workflow_str_list[i]["name"] == "erro_msg":
            success_rate = 0
            break     
        if workflow_str_list[i]["name"] == "code_execution_success":
            success_rate = 1
            break
        if workflow_str_list[i]["name"] == "code_execution_error":
            success_rate = 0
            break
    
    tc_dict["success_rate"] = success_rate
    
 
    tc_dict["all_step"] = plan_list
    all_step_num = len(plan_list)

    tc_dict["task_complete_rate"] = float(completed_step)/float(all_step_num)#task_complete
    tc_dict["step_number"] = all_step_num
    
    return tc_dict

def task_completion_autogen(workflow_str, eval_model="gpt-4o"):
    workflow_str_list = eval(workflow_str)
    plan_str = ""
    manager_str = []
    for i in workflow_str_list:
        if i["name"] == "planner":
            plan_str = i["content"]
        if i["name"] == "manager":
            tmp = i["content"]
            manager_str.append(i["content"])
    final_manager_str = []
    if "All steps completed. Invoking Terminator agent to TERMINATE the process" in tmp:
        for i in range(len(manager_str)):
           final_manager_str.append(manager_str[i])
    else:
        for i in range(len(workflow_str_list)-1,-1,-1):
            if workflow_str_list[i]["name"] == "executor":
                #if "exitcode: 1" not in workflow_str_list[i]["content"]:
                if "exitcode: 0" in workflow_str_list[i]["content"]:
                    for i in range(len(manager_str)):
                        final_manager_str.append(manager_str[i])
                    break
                else:
                    for i in range(len(manager_str)-1):
                        final_manager_str.append(manager_str[i])
                    break
    completed_steps = []
    #import pdb
    #pdb.set_trace()
    for i in final_manager_str:
        try:
            tmp_ = eval(i)["Current Step"]
            completed_steps.append(tmp_)
        except:
            pass
        
    #import pdb
    #pdb.set_trace()
    tc_dict = {}
    
    #with open(workflow_path) as f:
    #    workflow_data = f.read()
    config_list = CONFIG_LIST[eval_model]
    prompt = """ 
Given a bioinformatics analysis plan, please extract the steps and summarize them as a list. The output should be formatted as a JSON object containing numbered steps with their descriptions. Ensure the JSON structure includes keys like `step_1`, `step_2`, etc., and each step should have a clear description. Avoid any markdown formatting and ensure the JSON is valid. 
Example output format:  
```json  
{  
  "step_1": "Description of the first step",  
  "step_2": "Description of the second step",  
  ...  
}  
```  
Please make sure the steps are comprehensive and do not omit any critical parts of the plan.

I will provide you the plan information as the following:\n
"""+plan_str
    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']
    
    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )

    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        indic += 1
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        all_steps = json_str
    else:
        json_str = "".join(json_str.split("\n"))
        all_steps = eval(json_str)
    
    tc_dict["all_step"] = all_steps
    '''
    prompt = """ 
Given the all steps of one bioinformatics task and the completed steps by multi-agent system, please determine which steps have been completed. The output should be formatted as a JSON object where each key is the step number and the value is an object containing the step description and a boolean indicating whether it is completed.
Example output format (strictly in this format):
```json  
{  
  "step_1": {  
    "description": "Description of the first step",  
    "completed": "Yes"/"No"  
  },  
  "step_2": {  
    "description": "Description of the second step",  
    "completed": "Yes"/"No"  
  },  
  ...  
}
```  
Instructions:
- Match steps based on their descriptions, not just keywords.
- If a step from the completed steps by multi-agent system does not match any step in the analysis plan, ignore it.
- Ensure the output JSON is valid and does not contain any markdown formatting.
Please process the input and generate the output as specified.

I provide you with the all steps of one bioinformatics task as the following:\n
"""+str(all_steps)+"\n\n"+"I provide you with the completed steps by multi-agent system as the following:\n" + str(completed_steps)
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    import re
    json_str = ""
    
    while(json_str==""):
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        #if json_str == "":
        #    json_str = model_reply
    json_str = "".join(json_str.split("\n"))
    tmp = eval(json_str)
    '''
    all_step_num = len(all_steps)
    if len(completed_steps) > all_step_num:
        completed_step_num = all_step_num
    else:
        completed_step_num = len(completed_steps)
    #completed_step_num = 0
    #for key, values in tmp.items():
    #    #import pdb
    #    #pdb.set_trace()
    #    if values["completed"] == "Yes":
    #        completed_step_num += 1
    #import pdb
    #pdb.set_trace()
    tc_dict["task_complete_rate"] = float(completed_step_num)/float(all_step_num)#task_complete
    tc_dict["step_number"] = all_step_num

    return tc_dict

##---------------------Task Completion -----------------------------------
def task_completion(workflow_str, eval_model="gpt-4o"):
    tc_dict = {}
    #with open(workflow_path) as f:
    #    workflow_data = f.read()
    config_list = CONFIG_LIST[eval_model]

    prompt = """
You are tasked with evaluating the task completion rate of an intelligent agent system based on its workflow. Planner in the workflow provides a series of steps of the task that the agent must complete. The task completion rate is defined as the ratio of the steps completed by the agent to the total planned steps.

Please provide the following information in the specified format:

1. **Total Planned Steps**: The complete list of steps in the workflow that the intelligent agent is expected to complete.
2. **Completed Steps**: The list of steps that the intelligent agent has already completed.
3. **Task Completion Rate**: Calculate the task completion rate using the formula: (Number of Completed Steps / Total Planned Steps) * 100%.

**Format for Output** (strictly in json format):
{
  "Total Planned Steps": [List of all planned steps],
  "Completed Steps": [List of completed steps],
  "Task Completion Rate": [Calculated completion rate as a percentage]
}

**Example of Your Output** (strictly in json format):
```json
{
  "Total Planned Steps": ["Step 1: Initialize System", "Step 2: Collect Data", "Step 3: Process Data", "Step 4: Generate Report", "Step 5: Send Report"],
  "Completed Steps": ["Step 1: Initialize System", "Step 2: Collect Data"],
  "Task Completion Rate": "40%"
}
```json

Please strictly follow the output format.

I provide you the workflow of the multi-agent system:

"""+ workflow_str

    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )

    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        indic += 1
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        tmp = json_str
    else:
        json_str = "".join(json_str.split("\n"))
        tmp = eval(json_str)
    tc_dict["task_complete_rate"] = eval(tmp["Task Completion Rate"][0:-1])/100#task_complete
    tc_dict["step_number"] = len(tmp["Total Planned Steps"])
    tc_dict["all_step"] = tmp["Total Planned Steps"]

    return tc_dict
##---------------------Task Completion -----------------------------------

##---------------------Round for autogen-----------------------------------
def round_statics(workflow_list, step_number):
    all_rounds = len(workflow_list) - 1
    correct_rounds = 0
    for idx, agent in enumerate(workflow_list):
        if agent['name'] == 'executor':
            if "exitcode: 1" in agent['content']:
                correct_rounds += 1
    round_str = [all_rounds / float(step_number), correct_rounds / float(step_number)]
    #with open(round_path, 'w', encoding='utf-8') as f:
    #    f.write(str(round_str))
    #print(f"generate {model_name}/{tool_name} round txt")
    return round_str
##---------------------Round -----------------------------------

##---------------------Round for langgraph-----------------------------------
def round_statics_m2(workflow_list, step_number):
    all_rounds = len(workflow_list) - 1
    error_rounds = 0
    for idx, agent in enumerate(workflow_list):
        if agent['name'] == 'code_execution_error':
            error_rounds += 1
    round_str = [all_rounds / float(step_number), error_rounds / float(step_number)]
    #with open(round_path, 'w', encoding='utf-8') as f:
    #    f.write(str(round_str))
    #print(f"generate {model_name}/{tool_name} round txt")
    return round_str
##---------------------Round -----------------------------------

##---------------------RAG -----------------------------------
def rag_eval(rag_list, tool_name, database_path,eval_model="gpt-4o"):
    if rag_list == []:
        return 0.0, 0.0, [None, None, None]
    config_list = CONFIG_LIST[eval_model]
    retrieval_list = []
    new_rag_list = []
    for i in rag_list:
        
        tmp_ = {}
        tmp_['Current Step'] = i['Current Step']
        tmp_['RAG Required'] = i['RAG Required']
        new_rag_list.append(tmp_)
        if i['RAG Required'] == 'Yes':
            if "rag_results" in i:
                for j in i["rag_results"]:
                    try:
                        retrieval_list.append(j["retrieved_content"])
                    except:
                        pass
            else:
                try:
                    retrieval_list.append(i["retrieved_content"])
                except:
                    pass
    total_step = len(rag_list)
    retrieval_step = len(retrieval_list)
    retrieval_relate_step = 0
    
    #open database file
    if tool_name == "scgen-integration":
        with open(f"{database_path}/scgen.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    elif tool_name == "seurat-1" or tool_name=="seurat-2":
        with open(f"{database_path}/seurat.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    else:
        with open(f"{database_path}/{tool_name}.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    from fuzzywuzzy import fuzz
    for i in retrieval_list:
        similarity_score = fuzz.ratio(rag_ref, i)
        if similarity_score > 95:
            retrieval_relate_step += 1
    
    try:    
        retrieval_accuracy = float(retrieval_relate_step) / float(retrieval_step)
    except:
        retrieval_accuracy = 1.0
    prompt = """
You are an AI evaluator specialized in validating RAG decisions. Analyze each step and determine if the RAG Required flag was correctly set. Follow these **strict rules**:

【MANDATORY RAG TRIGGERS】 
1. **ANY explicit tool/system name** mentioned in step descriptions enclosed in angle brackets (e.g., "<Excel Data Analysis Tool>", "<Salesforce CRM>")
2. Access to external systems:
   - APIs (e.g., Google Maps API)
   - Calculation tools (e.g., Wolfram Alpha)
   - Specialized software (e.g., AutoCAD)

【GENERAL RAG CRITERIA】
3. Requires non-public data (internal docs/private knowledge bases)
4. Needs verifiable citations/sources
5. Involves confidential/personal data processing

【RAG NOT REQUIRED SCENARIOS】 
1. Common knowledge/logic reasoning (e.g., "calculate circle area")
2. General theory application (e.g., "apply SWOT analysis")
3. Subjective/creative tasks (e.g., "write a poem")

【VALIDATION WORKFLOW】 
1. FIRST check for <tool/system names> 
   - If present → RAG MUST be "Yes"
   - If absent → Apply general criteria
2. Match remaining conditions
3. Compare with agent's decision

【REQUIRED OUTPUT FORMAT】:
{
  "validation_results": [
    {
      "Step": "Original step text",
      "Tool_Detected": "True/False",
      "Expected_RAG": "Yes/No",
      "Agent_Decision": "Yes/No",
      "Is_Correct": "Yes/No",
      "Violation_Type": "None/Missing_RAG/Unnecessary_RAG" 
    },
    #here you need to generate result dict for each step in the provided RAG process.
    ...
  ],
  "accuracy_score": "X.X%"
}

Please strictly follow the output format.

I provide you the RAG process of the multi-agent system:

"""+ str(new_rag_list)

    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        indic += 1
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        out = json_str
    else:
        out = eval(json_str)
    
    if_retrieval_accuracy = out["accuracy_score"]
    return if_retrieval_accuracy, retrieval_accuracy, [total_step, retrieval_step, retrieval_relate_step]
##---------------------RAG -----------------------------------


def rag_eval_noplan(rag_list, tool_name, database_path, eval_model="gpt-4o"):
    config_list = CONFIG_LIST[eval_model]
    if rag_list == []:
        return 0.0, 0.0, [None, None, None]
    
    retrieval_list = []
    new_rag_list = []
    for i in rag_list:
        
        tmp_ = {}
        tmp_['Current Step'] = i['Current Step']
        tmp_['RAG Required'] = i['RAG Required']
        new_rag_list.append(tmp_)
        if i['RAG Required'] == 'Yes':
            if "rag_results" in i:
                for j in i["rag_results"]:
                    try:
                        retrieval_list.append(j["retrieved_content"])
                    except:
                        pass
            else:
                try:
                    retrieval_list.append(i["retrieved_content"])
                except:
                    pass
    total_step = len(rag_list)
    retrieval_step = len(retrieval_list)
    retrieval_relate_step = 0
    
    #open database file
    if tool_name == "scgen-integration":
        with open(f"{database_path}/scgen.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    elif tool_name == "seurat-1" or tool_name=="seurat-2":
        with open(f"{database_path}/seurat.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    else:
        with open(f"{database_path}/{tool_name}.md", "r", encoding="utf-8") as f:
            rag_ref = f.read()
    from fuzzywuzzy import fuzz
    for i in retrieval_list:
        similarity_score = fuzz.ratio(rag_ref, i)
        if similarity_score > 95:
            retrieval_relate_step += 1
    
    try:    
        retrieval_accuracy = float(retrieval_relate_step) / float(retrieval_step)
    except:
        retrieval_accuracy = 1.0
    prompt = """
You are an AI evaluator specialized in validating RAG decisions. Analyze each step and determine if the RAG Required flag was correctly set. Follow these **strict rules**:

【MANDATORY RAG TRIGGERS】 
1. **ANY explicit tool/system name** mentioned in step descriptions enclosed in angle brackets (e.g."<Retrieve ... >" ...)
2. Access to external systems:
   - Bioinformatic tools (eg. DESeq2, Doucling, Hotspot ect.)
   - APIs (e.g., Google Maps API)
   - Calculation tools (e.g., Wolfram Alpha)
   - Specialized software (e.g., AutoCAD)

【GENERAL RAG CRITERIA】
3. Requires non-public data (internal docs/private knowledge bases)
4. Needs verifiable citations/sources
5. Involves confidential/personal data processing

【RAG NOT REQUIRED SCENARIOS】 
1. Common knowledge/logic reasoning (e.g., "calculate circle area")
2. General theory application (e.g., "apply SWOT analysis")
3. Subjective/creative tasks (e.g., "write a poem")

【VALIDATION WORKFLOW】 
1. FIRST check for <tool/system names> 
   - If present → RAG MUST be "Yes"
   - If absent → Apply general criteria
2. Match remaining conditions
3. Compare with agent's decision

【REQUIRED OUTPUT FORMAT】:
{
  "validation_results": [
    {
      "Step": "Original step text",
      "Tool_Detected": "True/False",
      "Expected_RAG": "Yes/No",
      "Agent_Decision": "Yes/No",
      "Is_Correct": "Yes/No",
      "Violation_Type": "None/Missing_RAG/Unnecessary_RAG" 
    },
    #here you need to generate result dict for each step in the provided RAG process.
    ...
  ],
  "accuracy_score": "X.X%"
}

Please strictly follow the output format.

I provide you the RAG process of the agent system:

"""+ str(new_rag_list)
    print(f"new_rag_list: {new_rag_list}")
    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        print(f"model_reply: {model_reply}")
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        indic += 1
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        out = json_str
    else:
        out = eval(json_str)
    
    if_retrieval_accuracy = out["accuracy_score"]
    return if_retrieval_accuracy, retrieval_accuracy, [total_step, retrieval_step, retrieval_relate_step]

def consistency_with_plan(plan_str, code_file_path, eval_model="gpt-4o"):

    
    with open(code_file_path) as f:
        code_str = f.read()
    

    clean_code_str = ""
    for i in code_str.split("\n"):
        if i.startswith("#"):
            continue
        if i == "":
            continue
        clean_code_str += i + "\n"
    if clean_code_str == "":
        return {"score": 0, "evaluation_summary": "The code is empty or contains only comments.", "detailed_analysis": {"plan_coverage": "0 of plan elements implemented", "technical_alignment": "No technical alignment", "functional_completeness": "No functional completeness"}}

    config_list = CONFIG_LIST[eval_model]
    prompt = """
Act as an expert Code Implementation Auditor. Your task is to analyze given plans and corresponding code implementations, then evaluate how well the code executes the original plan. Follow these steps:

1. Carefully review the provided plan with its key requirements and objectives
2. Thoroughly examine the submitted code implementation
3. Perform a point-by-point comparison between plan specifications and code execution
4. Assess implementation completeness, logical alignment, and technical fidelity
5. Rate the implementation on a 0-5 scale using this rubric:
   - 5~4: Perfect alignment, almost all plan aspects implemented correctly
   - 4~3: Minor omissions/errors not affecting core functionality
   - 3~2: Partial implementation with some key elements missing
   - 2~1: Significant deviations affecting primary objectives
   - 1~0: Superficial implementation missing most requirements

Output a JSON object with this exact structure:
{
  "score": [0~5],
  "evaluation_summary": "[concise 2-3 sentence assessment]",
  "detailed_analysis": {
    "plan_coverage": "[xx of plan elements implemented]",
    "technical_alignment": "[analysis of architectural consistency]",
    "functional_completeness": "[assessment of requirement fulfillment]"
  }
}

The given plan as follows:""" + "\n" + plan_str + """\n\nThe corresponding code implementations as follows:""" + "\n" + code_str
    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        indic += 1
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        out = json_str
    else:
        out = eval(json_str)
    #import pdb
    #pdb.set_trace()
    
    return out

def consistency_with_plan_lang(workflow_str_list, plan_str, eval_model="gpt-4o"):
    
    config_list = CONFIG_LIST[eval_model]
    code_str = ""
    for i,_ in enumerate(workflow_str_list):
        if workflow_str_list[i]["name"] == "code_generate" and workflow_str_list[i]["content"] not in code_str:
            code_str += workflow_str_list[i]["content"]+"\n\n"
    if code_str == "":
        return {"score": 0, "evaluation_summary": "The code is empty or contains only comments.", "detailed_analysis": {"plan_coverage": "0 of plan elements implemented", "technical_alignment": "No technical alignment", "functional_completeness": "No functional completeness"}}


    prompt = """
Act as an expert Code Implementation Auditor. Your task is to analyze given plans and corresponding code implementations, then evaluate how well the code executes the original plan. Follow these steps:

1. Carefully review the provided plan with its key requirements and objectives
2. Thoroughly examine the submitted code implementation
3. Perform a point-by-point comparison between plan specifications and code execution
4. Assess implementation completeness, logical alignment, and technical fidelity
5. Rate the implementation on a 0-5 scale using this rubric:
   - 5: Perfect alignment, all plan aspects implemented correctly
   - 4: Minor omissions/errors not affecting core functionality
   - 3: Partial implementation with some key elements missing
   - 2: Significant deviations affecting primary objectives
   - 1: Superficial implementation missing most requirements
   - 0: Completely unrelated or non-functional code

Output a JSON object with this exact structure:
{
  "score": [0-5],
  "evaluation_summary": "[concise 2-3 sentence assessment]",
  "detailed_analysis": {
    "plan_coverage": "[xx of plan elements implemented]",
    "technical_alignment": "[analysis of architectural consistency]",
    "functional_completeness": "[assessment of requirement fulfillment]"
  }
}

The given plan as follows:""" + "\n" + str(plan_str) + """\n\nThe corresponding code implementations as follows:""" + "\n" + code_str
    import openai
    try:
        openai.api_type = config_list[0]['api_type']
    except:
        pass
    try:
        openai.api_version = config_list[0]['api_version']
        openai.azure_endpoint = config_list[0]['azure_endpoint']
    except:
        openai.base_url = config_list[0]['azure_endpoint']
    openai.api_key = config_list[0]['api_key']

    #response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model=config_list[0]['model'], # replace this value with the deployment name you chose when you deployed the associated model.
    messages = [{"role":"user", "content": prompt}],
    temperature=0,
    #max_tokens=2048,
    )
    import re
    json_str = ""
    indic = 0
    while(json_str==""):
        if indic > 3:
            break
        model_reply = response.choices[0].message.content
        #print(model_reply)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, model_reply, re.DOTALL)
        json_str = '\n'.join(matches)
        indic += 1
        if json_str == "":
            try:
                json_str = eval(model_reply)
            except:
                pass
        #if json_str == "":
        #    json_str = model_reply
    if type(json_str) == dict:
        out = json_str
    else:
        out = eval(json_str)
    #import pdb
    #pdb.set_trace()
    
    return out