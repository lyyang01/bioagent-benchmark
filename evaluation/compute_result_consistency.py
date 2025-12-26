import argparse
import json
import os
import numpy as np
import pandas as pd
import pyreadr
import scanpy as sc
from scib_metrics.benchmark import Benchmarker
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--frame_work", type=str, default="autogen")
parser.add_argument("--lab", type=str)

parser.add_argument("--prompt_path", type=str, default="../../prompt/prompt_gradient.json")
parser.add_argument("--result_path", type=str, default="../../logs/logs_by_workflow")
parser.add_argument("--gt_path", type=str, default="../../groundtruth_result")
parser.add_argument("--out_path", type=str, default="consistency_results")

args = parser.parse_args()

model_names = [args.model]
frame_work = args.frame_work
lab = args.lab
result_path = args.result_path
gt_path = args.gt_path
out_path = args.out_path


final_output_dict = {}
for model_name in model_names:
    #print(f"*************Begin eval on {model_name}***************")
    final_output_dict[model_name] = {}
    def find_files(start_path, prefix):
        matching_files = []
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if file.startswith(prefix):
                    matching_files.append(os.path.join(root, file))
        return matching_files

    root_path_ = f"{result_path}/{lab}/{frame_work}/{model_name}/"


    with open(args.prompt_path, "r") as f:
        tool_dict = json.load(f)

    tool_names = list(tool_dict.keys())

    success_generate_output = 0
    success_tool = []

    #if frame_work == "react":
    #    work_flow_path = root_path_ + f"react_{model_name}_scores.json"

    #else:
    work_flow_path = root_path_ + f"{frame_work}_{model_name}_scores.json"

    with open(work_flow_path, "r") as f:
        all_scores_ = json.load(f)

    #read all sucess_task_all
    #with open("/public/home/testdata/yangliu/agentbench/share_data/evaluation_scripts/success_task_all.json", "r") as f:
    #    all_success_tasks = json.load(f)

    for key in tool_names:
        try:
            if frame_work == "autogen":
                work_flow_path = root_path_ + f"{key}/{key}-workflow.txt"
                with open(work_flow_path, "r") as f:
                    work_flow_str = f.read()
                if "All steps completed. Invoking Terminator agent to TERMINATE the process." in work_flow_str:
                    success_tool.append(key)
                else:
                    work_flow_list = eval(work_flow_str)
                    for i in range(len(work_flow_list)-1,-1,-1):
                        if work_flow_list[i]["name"] == "executor":
                            if "exitcode: 1" not in work_flow_list[i]["content"]:
                                success_tool.append(key)
                                break
                            else:
                                break
            else:
                
                if all_scores_[key]["execution"]["success_rate"] == 1:
                    success_tool.append(key)
                #success_tool = all_success_tasks[frame_work][model_name]
        except:
            pass


    def cosine_similarity_my(v1, v2):
        
        if len(v1.shape) == 1:
            v1 = v1.reshape(v1.shape[0], 1)
        if len(v2.shape) == 1:
            v2 = v2.reshape(v2.shape[0], 1)
        #try:
        similarity_matrix = cosine_similarity(v1, v2)
        #except Exception as e:
        #    print(e); import pdb; pdb.set_trace()
        tmp = np.mean(similarity_matrix)
        if tmp < 0:
            return 0
        return np.mean(similarity_matrix)

    from scipy.spatial.distance import jensenshannon
    def _js_similarity(umap1, umap2, n_bins):
        """Jensen-Shannon散度相似性"""
        similarities = []
        
        # 对每个维度计算JS散度
        for dim in range(umap1.shape[1]):
            # 计算直方图
            min_val = min(umap1[:, dim].min(), umap2[:, dim].min())
            max_val = max(umap1[:, dim].max(), umap2[:, dim].max())
            
            hist1, _ = np.histogram(umap1[:, dim], bins=n_bins, range=(min_val, max_val), density=True)
            hist2, _ = np.histogram(umap2[:, dim], bins=n_bins, range=(min_val, max_val), density=True)
            
            # 添加小值避免零概率
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # 计算JS散度
            js_div = jensenshannon(hist1, hist2)
            
            # JS散度转换为相似性得分 (0-1)
            js_similarity = 1 - js_div
            similarities.append(js_similarity)
        
        return np.mean(similarities)

    def compare_vectors_by_clustering(vec1, vec2, n_clusters=5, pca_components=2):
        
        # 对每个向量进行聚类
        #kmeans1 = KMeans(n_clusters=n_clusters, random_state=42)
        #kmeans1.fit(vec1)
        
        #kmeans2 = KMeans(n_clusters=n_clusters, random_state=42)
        #kmeans2.fit(vec2)
        
        # 获取聚类中心
        #centers1 = kmeans1.cluster_centers_
        #centers2 = kmeans2.cluster_centers_
        
        # 对每个聚类中心分别进行 PCA 降维
        # 对 centers1 进行 PCA 降维
        if vec1.shape[1] == vec2.shape[1]:
            centers1_pca = vec1
            centers2_pca = vec2
        else:
            if vec1.shape[1] < vec2.shape[1]:
                pca_components = vec1.shape[1]
                
                centers1_pca = vec1
                # 对 centers2 进行 PCA 降维
                pca2 = PCA(n_components=pca_components)
                pca2.fit(vec2)
                centers2_pca = pca2.transform(vec2)
            else:
                pca_components = vec2.shape[1]
                centers2_pca = vec2

                # 对 centers2 进行 PCA 降维
                pca1 = PCA(n_components=pca_components)
                pca1.fit(vec1)
                centers1_pca = pca1.transform(vec1)
        #import pdb
        #pdb.set_trace()
        # 计算投影后的聚类中心之间的相似度
        # 这里使用余弦相似度，也可以选择其他相似度度量
        #similarity_matrix = cosine_similarity(centers1_pca, centers2_pca)
        similarity = _js_similarity(centers1_pca, centers2_pca, n_bins=50)

        # 取相似度矩阵的平均值作为最终相似度
        #similarity = np.mean(similarity_matrix)
        

        return similarity

    fail_save_output = 0
    fail_save_output_list = []
    print(success_tool)
    
    no_need_compute_consistency = []
    consistency_score = {}
    #debug_tool = "scvi"


    for key in success_tool:
        #if key not in [debug_tool]:
        #    continue
        
        #try:
            gold_root_path = gt_path
            file_names = find_files(gold_root_path, key+".")
            if len(file_names) == 0:
                no_need_compute_consistency.append(key)
                continue
            assert len(file_names) == 1
            if file_names[0].endswith("npy"):
                gold_vector = np.load(file_names[0])
            elif file_names[0].endswith("csv"):
                gold_vector = pd.read_csv(file_names[0]).to_numpy()
                if key in ["scvi", "scanvi", "harmony"]:
                    gold_vector = gold_vector[:, 1:]
                    gold_vector = gold_vector[0].astype(float)
                    
            elif file_names[0].endswith("RData"):
                tmp_ = pyreadr.read_r(file_names[0])
                try:
                    df = tmp_["tmp"]
                except:
                    df = tmp_["saved_data"]
                gold_vector = df.to_numpy()
            #print(gold_vector.shape)

            
            result_root_path = root_path_ + f"{key}/agent_output/"
            if key == "contrastivevi":
                file_names = find_files(result_root_path, key+".")
                if len(file_names) == 0:
                    file_names = find_files(result_root_path, "cellassign"+".")
                    if len(file_names) == 0:
                        print(f"no output files in the agentoutput of {key}!")
                        fail_save_output += 1
                        fail_save_output_list.append(key)
                        continue
            else:
                file_names = find_files(result_root_path, key+".")
                if len(file_names) == 0:
                    print(f"no output files in the agentoutput of {key}!")
                    fail_save_output += 1
                    fail_save_output_list.append(key)
                    continue

            print(f"eval on {key}")
            success_generate_output += 1

            #if (model_name == "grok3-beta") and (key == "scanorama"):
            #    tmp = file_names[0].split("/")[0:-1]
            #    tmp.append("scanorama.csv")
            #    file_names[0] = "/".join(tmp)

            try:
                assert len(file_names) > 0
            except:
                import pdb
                pdb.set_trace()
            result_path = file_names[0]
            #=======END find agent output file names========
            if file_names[0].endswith("csv"):
                try:
                    agent_vector = pd.read_csv(file_names[0]).to_numpy()
                except:
                    print(f"no output files in the agentoutput of {key}!")
                    continue
                
            elif file_names[0].endswith("RData"):
                #import pdb
                #pdb.set_trace()
                try:
                    tmp_ = pyreadr.read_r(file_names[0])
            
                    obj_name = list(tmp_.keys())
                    df = tmp_[obj_name[0]]
                except:
                    print(f"read RData error for {key}!")
                    continue
                
                agent_vector = df.to_numpy()
            elif file_names[0].endswith("h5ad"):
                adata = sc.read(file_names[0])
                #if (key == "harmony") and ((model_name == "deepseek-r1")) and (frame_work == "langgraph"):
                #    consistency_score[key] = 1.0
                #    continue

                bm = Benchmarker(
                        adata,
                        batch_key="batch",
                        label_key="cell_type",
                        embedding_obsm_keys=[f"X_{key}"],
                        n_jobs=-1,
                    )
                #except:
                #    import pdb
                #    pdb.set_trace()
                
                #    consistency_score[key] = 1.0
                bm.benchmark()
                df = bm.get_results(min_max_scale=False)
                agent_vector = df.to_numpy()
                if key in ["scvi", "scanvi", "harmony"]:
                    agent_vector = agent_vector[0].astype(float)
                #import pdb
                #pdb.set_trace()
            #=======END read agent output files=====
            
            #gene_list
            if (key == "singlecellhaystack") or (key == "scorpius") or (key == "scry") or (key == "scpnmf"):
                try:
                    if agent_vector.shape[0] >= gold_vector.shape[0]:
                        agent_vector = agent_vector[0:gold_vector.shape[0],:]
                    #else:
                    #    gold_vector = gold_vector[0:agent_vector.shape[0],:]
                    correct = 0
                    for i in agent_vector:
                        if i in gold_vector:
                            correct += 1
                        
                    consistency_score[key] = float(correct) / gold_vector.shape[0]
                    continue
                except:
                    continue
            #gene_list
            elif (key == "deeptree"):
                correct = 0
                exist_tmp = []
                for i in gold_vector:
                #for i in agent_vector:
                    if i in agent_vector:
                        correct += 1
            #gene list
            elif (key == "decoupler") or (key=="cellassign") or (key=="celltypist") or (key=="graphst") or (key=="doucling"):
                correct = 0
                exist_tmp = []
                try:
                    for i in gold_vector:
                    #for i in agent_vector:
                        try:
                            if i in agent_vector:
                                correct += 1
                        except:
                            agent_vector = agent_vector[:, 0:-1]
                            if i in agent_vector:
                                correct += 1
                            #if i.tolist()[0] not in exist_tmp:
                            #    exist_tmp.append(i.tolist()[0])
                            #    correct += 1
                    consistency_score[key] = float(correct) / gold_vector.shape[0]
                except:
                    print(f"fail to compute score for {key}")
                    continue
                assert consistency_score[key] <= 1
                #import pdb
                #pdb.set_trace()
                continue
            
            elif (key == "resolvi") or (key == "spage") or (key == "spagcn"):
                #import pdb
                #pdb.set_trace()
                try:
                    agent_vector = agent_vector[:, 1]
                except:
                    print(f"fail to compute score for {key}")
                    continue
                gold_vector = gold_vector[:, 1]
            
            else:
                try:
                    #change agent_vector to float
                    index = (0,) * agent_vector.ndim
                    if type(agent_vector[index]) == str:
                        agent_vector = agent_vector[:,1:]
                    if agent_vector.dtype == "O":
                        agent_vector = agent_vector.astype(float)
                    #change gold_vector to float
                    if type(gold_vector[index]) == str:
                        gold_vector = gold_vector[:,1:]
                    if gold_vector.dtype == "O":
                        gold_vector = gold_vector.astype(float)
                except:
                    print(f"fail to compute score for {key}")
                    continue
            #import pdb
            #pdb.set_trace()
            try:
                #import pdb
                #pdb.set_trace()
                if agent_vector.shape == gold_vector.shape:
                    consistency_score[key] = cosine_similarity_my(gold_vector, agent_vector)
                else:
                    ##
                    #if (key == "paga") or (key == "cellrank") or (key == "scvelo"):
                    #    print(f"no equal shape for {key}")
                    #    continue
                    #else:
                        consistency_score[key] = compare_vectors_by_clustering(gold_vector, agent_vector)
            except:
                print(f"fail to compute score for {key}")
                continue
            
            try:
                assert consistency_score[key] <= 1
            except:
                del consistency_score[key]
                print(f"fail to compute score for {key}")
            
            
    print("\n\n")
    print("fail save file num:", fail_save_output)
    print("all passing tool:", len(success_tool))
    print("can compute consistency tool:", len(consistency_score))
    print("no need compute consistency tool:", len(no_need_compute_consistency))
    print("\n\n")
    print(consistency_score)
    all_con_score = 0
    for value in consistency_score.values():
        all_con_score += value
    avg_con_score = all_con_score/float(len(consistency_score))
    #import pdb
    #pdb.set_trace()
    print("consistency score:", avg_con_score)
    avg_con_score2 = all_con_score/float(len(success_tool) - fail_save_output - len(no_need_compute_consistency))
    print("consistency score with all tool that saved files:", avg_con_score2)
    avg_con_score3 = all_con_score/float(44)
    print("consistency score with all tool:", avg_con_score3)
    print("\n\n")
    filter_success_tool = []
    for i in success_tool:
        if i in fail_save_output_list:
            pass
        else:
            filter_success_tool.append(i)
    print(filter_success_tool)

    #final_output_dict = {}
    final_output_dict[model_name]["consistency_dict"] = consistency_score
    final_output_dict[model_name]["fail_save_file_num"] = fail_save_output
    final_output_dict[model_name]["all_passing_tool"] = len(success_tool)
    final_output_dict[model_name]["can_compute_consistency_tool"] = len(consistency_score)
    final_output_dict[model_name]["no_need_compute_consistency_tool"] = len(no_need_compute_consistency)
    final_output_dict[model_name]["consistency_score"] = avg_con_score
    #final_output_dict["consistency_score_with_all_tool_that_saved_files"] = #avg_con_score2
    final_output_dict[model_name]["consistency_score_with_all_tool"] = avg_con_score3

with open(f"{out_path}/{frame_work}_{model_name}_consistency.json", "w") as f:
    json.dump(final_output_dict, f)
#import pdb
#pdb.set_trace()
