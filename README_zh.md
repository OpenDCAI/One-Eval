<div align="center">
  <!-- TODO: 在这里放项目 Logo -->
  <img src="./static/logo/logo.png" width="360" alt="One-Eval Logo" />

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/OpenDCAI/One-Eval)](./LICENSE)
[![Repo Size](https://img.shields.io/github/repo-size/OpenDCAI/One-Eval?color=green)](https://github.com/OpenDCAI/One-Eval)
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2603.09821)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/One-Eval)
[![WeChat Group](https://img.shields.io/badge/WeChat-Group-brightgreen?logo=wechat&logoColor=white)](./static/logo/318QRcode.jpg)

</div>

One-Eval 是一个面向大模型的自动化评测框架，核心目标是实现 **NL2Eval**：一句话从用户需求到优雅的模型评测报告。  
项目基于 [DataFlow](https://github.com/OpenDCAI/DataFlow) 与 [LangGraph](https://github.com/langchain-ai/langgraph) 构建，用最简单直观的方式帮助你优雅地完成模型评测。

[English](./README.md) | 简体中文

## 📰 1. 最新动态

- **[2026-03] 🎉 One-Eval (v0.1.0) 正式开源！**  
  我们发布了首个版本，支持从自然语言到评测报告的全链路自动化 (NL2Eval)。告别繁琐的手动脚本，让大模型评测像聊天一样简单、直观、可控。欢迎 Star 🌟 关注！

## 💡 为什么选择 One-Eval？

以往的模型评测框架通常需要用户自行寻找 Benchmarks、下载数据，并手动填写大量配置参数。  
**One-Eval** 旨在改变这一现状：**凡是能自动做到的，都将交给 Agent 自动完成**。从基准推荐到模型评测，我们致力于提供最直接、直观的评测体验。

## 🔍 2. 项目概览

传统评测往往面临脚本繁杂、流程割裂、难以复用的痛点。One-Eval 将评测重构为**图化执行过程 (Graph / Node / State)**，致力于打造下一代交互式评测体验：

- 🗣️ **NL2Eval**: 只需输入一段自然语言目标（例如“评估模型在数学推理任务上的表现”），系统自动解析意图并规划执行路径。
- 🧩 **全链路自动化**: 自动完成基准推荐、数据准备、推理执行、指标匹配、打分与多维度报告生成。
- ⏸️ **人机交互**: 支持关键节点（如基准选择、结果复核）的中断与人工干预，便于根据反馈实时调整评测策略。
- 📊 **可扩展架构**: 基于 DataFlow 的算子体系与 LangGraph 的状态管理，轻松集成私有数据集与自定义指标。

<!-- TODO: 在这里放 One-Eval 框架原理图 -->
![One-Eval Framework](./static/logo/eval_framework.png)

## ⚡ 3. 快速开始

### 3.1 安装环境（推荐方式）

提供了 Conda 与 uv 两种环境管理方式，任选其一即可快速上手：

#### 方式 A: Conda
```bash
conda create -n one-eval python=3.10 -y
conda activate one-eval
pip install -e .
```

#### 方式 B: uv
```bash
uv venv
uv pip install -e .
```

### 3.2 启动服务

One-Eval 采用前后端分离架构，请分别启动后端 API 与前端界面。

#### ① 启动后端 (FastAPI)
```bash
uvicorn one_eval.server.app:app --host 0.0.0.0 --port 8000
```

#### ② 启动前端 (Vite + React)
```bash
cd one-eval-web
npm install
npm run dev
```
访问 http://localhost:5173 即可开始交互式评测。
> 启动后应先进入设置界面，配置API、模型以及HF Token等参数（以支持批量下载数据），并点击保存。

### 3.3 极简代码模式 (开发者模式)

如果你更喜欢在代码中直接调用，可以直接运行内置的完整工作流示例：  
[workflow_all.py](./one_eval/graph/workflow_all.py)

```bash
# 示例：直接通过命令行发起一次数学能力评估
python -m one_eval.graph.workflow_all "我想评估我的模型在Reasoning上的表现"
```

该 Graph 展示了从 Query 解析到报告生成的完整闭环，欢迎基于此进行二次开发与节点扩展。

## 🗂️ 4. 评测基准库 (Bench Gallery)

One-Eval 内置了丰富的 **Bench Gallery**，用于统一管理各类评测基准的元信息（如任务类型、数据格式、Prompt 模板）。

> 目前已涵盖主流纯文本能力维度（无需复杂沙盒环境）：
> - 🧮 **Reasoning**: MATH, GSM8K, BBH, AIME...
> - 🌐 **General Knowledge**: MMLU, CEval, CMMLU...
> - 🔧 **Instruction Following**: IFEval...

![Bench Gallery](./static/logo/gallery.png)

## 📮 5. 联系与引用

如果您对本项目感兴趣，或有任何疑问与建议，欢迎通过 Issue 或加入微信群与我们联系。

•	📮 [GitHub Issues](../../issues)：提交 Bug 或功能建议。

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进。

<div align="center">
  <img src="./static/logo/318QRcode.jpg" width="30%">
</div>

如果您在研究中使用了 One-Eval，欢迎引用我们的工作：


## Citation
```bibtex
@misc{shen2026oneevalagenticautomatedtraceable,
      title={One-Eval: An Agentic System for Automated and Traceable LLM Evaluation}, 
      author={Chengyu Shen and Yanheng Hou and Minghui Pan and Runming He and Zhen Hao Wong and Meiyi Qiang and Zhou Liu and Hao Liang and Peichao Lai and Zeang Sheng and Wentao Zhang},
      year={2026},
      eprint={2603.09821},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.09821}, 
}
@article{liang2025dataflow,
  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
  journal={arXiv preprint arXiv:2512.16676},
  year={2025}
}
```

