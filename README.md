# 讯飞文本分类项目（xfyun-words-classification）

## 项目简介
本项目旨在对中文文本进行多类别分类，支持多种特征工程与模型方案，包括TF-IDF、词向量（Qwen3-Embedding）、BERT等，适用于文本分类任务的快速实验与对比。

## 目录结构
- `main.py`：主程序入口，支持多模型方案的训练与预测。
- `src/`：核心代码目录
  - `utils.py`：通用工具、基类、QwenEmbedder向量化等
  - `dl/bert_base.py`：BERT系列深度学习分类器
  - `tfidf/`：TF-IDF特征下的多种分类器（SVC、Logistic、RF、Voting等）
  - `llm/`：大模型相关代码
- `data/`：数据目录
  - `dataset/`：包含`train_all.csv`（训练集）、`test_text.csv`（测试集）
  - `result/`：存放预测结果
- `ckpts/`：模型检查点
- `config/`：配置文件目录
- `wandb/`：训练日志与可视化

## 安装依赖
建议使用Python 3.12+，推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境，安装依赖：
```bash
uv sync
```

## 数据说明
- 训练集：`data/dataset/train_all.csv`，包含`文本`和`类别`字段
- 测试集：`data/dataset/test_text.csv`，包含`文本`字段
- 预测结果输出至`data/result/`目录

## 支持的模型方案
- **TF-IDF + SVC/Logistic/RF/Voting**：传统机器学习方法，基于分词与TF-IDF特征
- **Qwen3-Embedding + SVC**：调用Qwen3-Embedding-0.6B API进行文本向量化，SVC分类
- **BERT（MacBERT等）**：基于transformers的深度学习文本分类

## 快速开始
1. 准备好数据集，放置于`data/dataset/`
2. 安装依赖
3. 运行主程序：
```bash
python main.py
```
4. 预测结果与F1分数将输出到`data/result/`目录

## 主要依赖
- pandas, numpy, scikit-learn, jieba
- transformers, datasets, torch
- openai（Qwen3-Embedding API）
- wandb（可选，训练可视化）

## 训练与预测流程
- 支持多模型方案自动对比，主程序会依次输出各方案的F1分数与预测结果
- BERT方案支持断点续训与wandb日志

## 贡献者
- ticoAg（1627635056@qq.com）

## 联系方式
如有问题或建议，欢迎联系贡献者邮箱。
