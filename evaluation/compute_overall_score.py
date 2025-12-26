import math
import os

def plan_score(x):
    return 0.4*(x/5)

def code_score(x, ast, rouge, consistency):
    return 0.6*(0.25*(x/5 + ast + rouge + consistency/5))

import math

def time_score1(x, max_time=10800, steepness=0.0003, midpoint=800):
    """
    改进的时间评分函数
    :param x: 实际时间(秒)
    :param max_time: 最大时间(默认10800秒)
    :param steepness: 陡峭度，控制曲线形状
    :param midpoint: 中点位置，控制曲线拐点
    """
    if x < 0: x = 0
    if x > max_time: x = max_time
    
    # 使用sigmoid函数的变体来创建平缓下降曲线
    normalized_x = x / max_time
    score = 1 / (1 + math.exp(steepness * (x - midpoint)))
    
    # 调整到精确的0-1范围
    min_score = 1 / (1 + math.exp(steepness * (max_time - midpoint)))
    max_score = 1 / (1 + math.exp(steepness * (0 - midpoint)))
    
    # 线性映射到[0,1]
    final_score = (score - min_score) / (max_score - min_score)
    return final_score

def time_score2(x, max_time=10800, base=1.8, offset=100):
    """优化版时间评分函数
    :param x: 实际时间(秒)
    :param base: 对数底数(1.5-2.5之间调整曲线)
    :param offset: 偏移量(防止0值问题)
    """
    if x < 0: x = 0
    if x > max_time: x = max_time
    
    # 自定义对数计算
    log_b = lambda val: math.log(val) / math.log(base)
    
    denominator = log_b(max_time + offset)
    numerator = log_b(x + offset)
    
    return 1 - numerator / denominator

def knowledge_score(x,y):
    return (x+y)*0.5

def task_completion_quality(x, passing, consistency, success):
    return (0.4*success + 0.3* consistency + 0.3 *(passing*0.5 + x*0.5))


def resouce_score(x, max_cpu=10, k=0.03, m=1.5):
    """
    CPU占用率评分函数
    :param x: CPU占用率 (0-10)
    :param k: 衰减系数 (默认0.03)
    :param m: 低值区增强指数 (默认1.5)
    """
    # 边界处理
    if x < 0: x = 0
    if x > max_cpu: x = max_cpu
    
    # 计算最大值对应的指数
    max_exp = k * (max_cpu ** m)
    
    # 计算当前值的指数
    current_exp = k * (x ** m)
    
    # 计算原始分数
    raw_score = math.exp(-current_exp)
    
    # 最小值对应的分数
    min_score = math.exp(-max_exp)
    
    # 归一化到0-1范围
    score = (raw_score - min_score) / (1 - min_score)
    
    return round(score, 3)


import math

def interaction_score(x, max_val=50, k=0.005, m=1.5):
    """
    交互轮次评分函数 (0-50轮)
    :param x: 实际交互轮次
    :param k: 衰减系数(默认0.005)
    :param m: 低值区增强指数(默认1.5)
    """
    return _score_calculation(x, max_val, k, m)

def correction_score(x, max_val=10, k=0.05, m=1.8):
    """
    纠错轮次评分函数 (0-10轮)
    :param x: 实际纠错轮次
    :param k: 衰减系数(默认0.05)
    :param m: 低值区增强指数(默认1.8)
    """
    return _score_calculation(x, max_val, k, m)

def _score_calculation(x, max_val, k, m):
    """统一评分计算核心"""
    # 边界处理
    if x < 0: x = 0
    if x > max_val: x = max_val
    
    # 计算最大值对应的指数
    max_exp = k * (max_val ** m)
    
    # 计算当前值的指数
    current_exp = k * (x ** m)
    
    # 计算原始分数
    raw_score = math.exp(-current_exp)
    
    # 最小值对应的分数
    min_score = math.exp(-max_exp)
    
    # 归一化到0-1范围
    score = (raw_score - min_score) / (1 - min_score)
    
    return round(score, 3)

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--result_json", type=str, default="mainresult.json")
parser.add_argument("--out_path", type=str, default="./output/")
parser.add_argument("--frame_work", type=str, default="autogen")
parser.add_argument("--model_id", type=int)
parser.add_argument("--w1", type=float, default=0.15)
parser.add_argument("--w2", type=float, default=0.15)
parser.add_argument("--w3", type=float, default=0.2)
parser.add_argument("--w4", type=float, default=0.5)

args = parser.parse_args()

path = args.result_json
out_path = args.out_path
frame_work = args.frame_work
model_id = args.model_id

if not os.path.exists(out_path):
    os.makedirs(out_path)


output_file = os.path.join(out_path, f"overall_score_{args.frame_work}.json")

with open(path, "r") as f:
    all_results = json.load(f)

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        final = json.load(f)
else:
    final = {}

model_names = [args.model]
final_dict = {}
for name in model_names:
    final_dict[name] = []

all_list = []
for key, value in all_results[frame_work].items():
    #for idx, i in enumerate(value):
    final_dict[model_names[0]].append(value[model_id])


w1=args.w1
w2=args.w2
w3=args.w3
w4=args.w4


for key, value in final_dict.items():
    plan_score_ = plan_score(value[2])
    code_score_ = code_score(value[3], value[4], value[5], value[6])
    time_score_ = time_score1(value[7])
    resouce_score_ = resouce_score(value[8])
    total_round_ = interaction_score(value[9], )
    correct_round_ = correction_score(value[10], )
    knowledge_score_ = knowledge_score(value[11], value[12])
    quality_score_ = task_completion_quality(value[13], value[14], value[15], value[16])
    overall_score = w1*(plan_score_+ code_score_) + w2*((time_score_+resouce_score_+total_round_+correct_round_)/4) + w3*knowledge_score_+w4*quality_score_
    final[key] = overall_score

with open(output_file, "w") as f:
    json.dump(final, f)
#print(final)
#import pdb
#pdb.set_trace()


