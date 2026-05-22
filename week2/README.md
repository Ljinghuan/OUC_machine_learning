# Week2 — 商场客户聚类与营销策略

## 作业要求对照

| 任务 | 实现 |
|------|------|
| 任务1 EDA | `run_analysis.py` / notebook：规模、缺失、异常值、直方图/箱线图/散点图 |
| 任务2 聚类 | 年收入 + 消费评分，K-Means，手肘法+轮廓系数，k=5 |
| 任务3 画像 | `output/cluster_profiles.csv` + 报告文字描述 |
| 任务4 营销 | `output/marketing_strategies.csv` + 报告第五章 |
| 任务5 提交 | `工程实训作业报告_week2.md` + `.ipynb` + `.py` |

## 快速运行

```powershell
conda activate ml
cd d:\OUC_machine_learning\week2
python run_analysis.py
```

## 导出 PDF 报告

在 VS Code / Typora 中打开 `工程实训作业报告_week2.md` 导出 PDF，或使用：

```powershell
# 若已安装 pandoc
pandoc 工程实训作业报告_week2.md -o 工程实训作业报告_week2.pdf
```

## 文件结构

```
week2/
├── Mall_Customers.csv
├── mall_customer_clustering.ipynb
├── run_analysis.py
├── 工程实训作业报告_week2.md
├── figures/          # 8 张图
└── output/           # 聚类结果与统计表
```
