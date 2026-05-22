# Week 1 — 信用卡违约预测（Kaggle / UCI 数据集）

基于 [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)（UCI 镜像）的机器学习分析，对应 notebook：`credit_card_default_kaggle_analysis.ipynb`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `credit_card_default_kaggle_analysis.ipynb` | 完整分析 notebook（EDA、采样、三模型对比、测试集评估） |
| `download_data.py` | 从 UCI 下载数据并保存为 `UCI_Credit_Card.csv` |
| `run_analysis.py` | 与 notebook 等价的可执行脚本，输出图表与 `model_comparison.csv` |
| `UCI_Credit_Card.csv` | 数据集（30,000 × 25，含目标列） |
| `figures/` | 运行后生成的可视化结果 |
| `model_comparison.csv` | 交叉验证模型对比表 |

## 环境

```powershell
conda activate ml
pip install -r week1/requirements.txt
```

## 运行方式

**Notebook（推荐）**

```powershell
cd d:\OUC_machine_learning\week1
jupyter notebook credit_card_default_kaggle_analysis.ipynb
```

**命令行脚本**

```powershell
cd d:\OUC_machine_learning\week1
python download_data.py   # 若尚无 CSV
python run_analysis.py
```

## 分析流程概要

1. 数据加载与清洗（删除 ID、统一 EDUCATION/MARRIAGE 编码）
2. EDA：目标分布、额度分布、相关性热力图
3. 特征工程：数值标准化 + 分类 one-hot
4. 模型：逻辑回归、随机森林、SVM；采样：无 / SMOTE / 欠采样
5. 5 折分层交叉验证，按 F1、ROC-AUC 选最优组合
6. 测试集报告、混淆矩阵、ROC / PR 曲线
