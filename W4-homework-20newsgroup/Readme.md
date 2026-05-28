### 一、数据集概述
**20 Newsgroups** 是经典的文本分类标准数据集，由 Ken Lang 于 1995 年收集并发布。
- 规模：约 **20,000 篇**英文新闻组文档，实际常用版本含 **18,846 篇**。
- 类别：**20 个主题**，分布基本均衡。
- 用途：广泛用于**文本分类、聚类、信息检索**等 NLP 任务的算法对比与基准测试。

### 二、20 个主题类别（分层命名）
- **comp.*（计算机）**
  - comp.graphics（图形）
  - comp.os.ms-windows.misc（Windows 杂项）
  - comp.sys.ibm.pc.hardware（IBM PC 硬件）
  - comp.sys.mac.hardware（Mac 硬件）
  - comp.windows.x（X Window）
- **rec.*（娱乐/运动）**
  - rec.autos（汽车）
  - rec.motorcycles（摩托车）
  - rec.sport.baseball（棒球）
  - rec.sport.hockey（冰球）
- **sci.*（科学）**
  - sci.crypt（加密）
  - sci.electronics（电子）
  - sci.med（医学）
  - sci.space（太空）
- **misc.*（杂项）**
  - misc.forsale（出售）
- **talk.*（辩论）**
  - talk.politics.misc（政治杂项）
  - talk.politics.guns（枪支）
  - talk.politics.mideast（中东）
  - talk.religion.misc（宗教杂项）
- **alt.* / soc.*（社会/宗教）**
  - alt.atheism（无神论）
  - soc.religion.christian（基督教）

### 三、常见版本
1. **原始版（20news-19997）**：含全部文档与原始头部信息，有重复。
2. **按时间划分版（20news-bydate）**：
   - 训练集：60%（时间较早）
   - 测试集：40%（时间较晚）
   - 已去重、移除部分头部，**推荐用于基准测试**。
3. **精简版（20news-18828）**：仅保留“发件人”与“主题”头部，去重。

### 四、数据特点
- **主题相似度可控**：既有强相关类（如 PC 硬件 vs Mac 硬件），也有完全无关类（如出售 vs 基督教），适合测试细粒度区分能力。
- **文本噪声真实**：含邮件头部、签名、引用文本、拼写错误、缩写，接近真实 NLP 场景。

### 五、在 Scikit-learn 中使用
```python
from sklearn.datasets import fetch_20newsgroups

# 加载数据（移除头部、签名、引用）
newsgroups = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    categories=["comp.graphics", "sci.med"]  # 可选指定类别
)

print(f"文档数：{len(newsgroups.data)}")
print(f"类别：{newsgroups.target_names}")
```

### 六、典型应用场景
- 文本分类算法对比（朴素贝叶斯、SVM、逻辑回归、BERT 等）
- 主题模型（LDA、PLSA）效果评估
- 文本表示学习（TF-IDF、Word2Vec、Doc2Vec）基准测试
