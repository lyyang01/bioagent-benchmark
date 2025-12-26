import json
import numpy as np

with open("/public/home/yangliu/workspaces/agent-benchmark-eval/mainresult-crossjudge.json", "r") as f:
    data = json.load(f)

frame_work = ["autogen", "langgraph", "react"]
eval_model = ["gpt-4o", "grok3-beta", "gemini-2.5-pro"]
key_metric = ["Content", "Plan\nAttributes", "Overall", "Code\nAttributes", "Consistency\nwith Plan", "Evoke\nAccuracy"]

final_dict = {}
#compute average
for i in frame_work:
    final_dict[i] = {}
    for j in key_metric:
        tmp = np.zeros(len(data["gpt-4o"][i][j]))

        for k in eval_model:
            tmp = np.array(data[k][i][j]) + tmp
        tmp = tmp/3

        final_dict[i][j] = tmp.tolist()

with open("mainresult-crossjudge-2.json", "w") as f:
    json.dump(final_dict, f)


with open("/public/home/yangliu/workspaces/agent-benchmark-eval/mainresult.json", "r") as f:
    ori_data = json.load(f)

for key in frame_work:
    value = ori_data[key]
    for subkey, subvalue in value.items():
        if subkey not in final_dict[key]:
            final_dict[key][subkey] = subvalue

with open("mainresult-crossjudge-final.json", "w") as f:
    json.dump(final_dict, f)

