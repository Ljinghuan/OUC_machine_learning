# W5 作业：二手车价格预测

## 文件

| 文件 | 说明 |
|------|------|
| `used_car_price_prediction.ipynb` | **主提交**：完整 Notebook（9 步实验流程） |
| `run_homework.py` | 脚本版（快速复现指标） |
| `data/car.csv` | 数据集（5512 条） |
| `figures/` | EDA 与预测图（运行 Notebook 后生成） |
| `output/model_metrics.csv` | 模型对比指标 |

## 运行

```powershell
conda activate ml
cd d:\OUC_machine_learning\W5_homework
pip install -r requirements.txt
jupyter notebook used_car_price_prediction.ipynb
```

或：`python run_homework.py`

## 实验概要

- **目标**：预测 `price_lakh`（价格，单位 Lakh）
- **特征**：品牌、里程、燃料、变速箱、过户次数、年份、排量、座位、车龄
- **模型**：随机森林（传统 ML）、MLP（神经网络）
- **指标**：MAE、MSE、RMSE、R²、MAPE
