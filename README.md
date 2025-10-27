# Benchmarking LLM Agents for Single-cell Omics Analysis

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.13201) [![Dataset](https://img.shields.io/badge/Dataset-Available-blue)](https://claude.ai/chat/89eaffbd-376e-46dc-bdf7-2db7db858d04#dataset) [![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data for **"Benchmarking LLM Agents for Single-cell Omics Analysis"**, a comprehensive evaluation framework designed to systematically assess the capabilities of Large Language Model (LLM)-based agents in single-cell omics analysis tasks.

![Framework Overview](./asserts/framework.png)<p align="center">
Fig1.Overview of the benchmarking framework architecture
</p>

## 🔬 Research Highlights

- **Comprehensive Benchmarking Framework**: A systematic evaluation system with 4 interrelated components for assessing LLM agent performance
- **Multi-Agent Support**: Evaluation across ReAct, LangGraph, and AutoGen frameworks
- **Extensive Model Coverage**: Assessment of 8 state-of-the-art LLMs including GPT-4o/4.1, Claude 3.7 Sonnet, DeepSeek R1/v3, Grok3, Gemini 2.5, and Qwen 2.5
- **Curated Task Dataset**: 50 representative single-cell omics analysis tasks with gold-standard outputs
- **17 Quantitative Metrics**: Comprehensive evaluation across cognitive synthesis, task quality, knowledge integration, and collaborative efficiency


## 📊 Dataset

**Download Dataset**: All datasets used in our benchmark are published at https://doi.org/10.5281/zenodo.17291196

We add instructions for the folders in the published dataset as the following:

- agent_benchmark

Datasets used in the benchmark. 'main' contains 50 tasks/tools for the main results. 'multiple-datasets' contains 13 tasks/tools and each task/tool implement on two different datasets. Here, 'data1' from 'main' and 'data2' is another different dataset.

- database

The tool base used in the benchmark. All 50 tools' documents are included. These documents contain how to use functions of different tools.

- groundtruth_code

The notebook and py/R script written by human researchers. They will be regarded as groundtruth codes during code evaluation.

- groundtruth_result

Saved results from codes written by human researchers. These results can be h5ad/csv/npy/RData types and will be regarded as groundtruth results during result consistency evaluation.

- input_prompt

Input prompts used in agents. 50 prompts for 'main' test and 13*2 prompts for 'multiple-datasets' test. Besides, for "main", gradient prompts are in the json file as well.

## 🏗️ LLM Configuration



## 🚀 Quick Start

*(to be updated soon)*


## 📄 Citation

If you use this framework or dataset in your research, please cite our paper:

```bibtex
@article{author2025benchmarking,
  title={Benchmarking LLM Agents for Single-cell Omics Analysis},
  author={Author, First and Author, Second and Author, Third},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```


## 📋 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.


---

**Keywords**: Large Language Models, LLM Agents, Single-cell Omics, Benchmarking, Computational Biology, scRNA-seq, Spatial Transcriptomic
