"""
商场客户聚类分析 — Week2 作业脚本
数据集: Mall_Customers.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

WEEK2 = Path(__file__).resolve().parent
DATA_PATH = WEEK2 / "Mall_Customers.csv"
FIG_DIR = WEEK2 / "figures"
OUT_DIR = WEEK2 / "output"

INCOME_COL = "Annual Income (k$)"
SPEND_COL = "Spending Score (1-100)"
FEATURES = [INCOME_COL, SPEND_COL]
K_RANGE = range(2, 11)
K_SUGGEST = range(4, 7)  # 作业建议 4～6 类


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def task1_eda(df: pd.DataFrame) -> dict:
    """任务1：数据探索"""
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "describe": df.describe(include="all").to_string(),
    }

    # 异常值：IQR 法
    outliers = {}
    for col in ["Age", INCOME_COL, SPEND_COL]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < low) | (df[col] > high)
        outliers[col] = int(mask.sum())
    info["outliers_iqr"] = outliers

    # 描述统计
    stats = df.groupby("Gender")[["Age", INCOME_COL, SPEND_COL]].agg(["mean", "std", "min", "max"])
    stats.to_csv(OUT_DIR / "eda_gender_stats.csv")

    # --- 可视化 ---
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 直方图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), ["Age", INCOME_COL, SPEND_COL, "CustomerID"]):
        if col == "CustomerID":
            continue
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_histograms.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 性别分布 + 箱线图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.countplot(data=df, x="Gender", ax=axes[0], palette="Set2")
    axes[0].set_title("Gender Distribution")
    sns.boxplot(data=df, x="Gender", y="Age", ax=axes[1], palette="Set2")
    axes[1].set_title("Age by Gender")
    sns.boxplot(data=df, x="Gender", y=SPEND_COL, ax=axes[2], palette="Set2")
    axes[2].set_title("Spending Score by Gender")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_gender_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=df, y=INCOME_COL, ax=axes[0], color="coral")
    axes[0].set_title("Annual Income")
    sns.boxplot(data=df, y=SPEND_COL, ax=axes[1], color="seagreen")
    axes[1].set_title("Spending Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_income_spend_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x=INCOME_COL, y=SPEND_COL, hue="Gender", ax=axes[0], alpha=0.8)
    axes[0].set_title("Income vs Spending (by Gender)")
    sns.scatterplot(data=df, x="Age", y=SPEND_COL, hue=INCOME_COL, palette="viridis", ax=axes[1], alpha=0.8)
    axes[1].set_title("Age vs Spending (color=Income)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_scatterplots.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 相关性
    num = df[["Age", INCOME_COL, SPEND_COL]].copy()
    num["Gender_num"] = LabelEncoder().fit_transform(df["Gender"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()

    return info


def task2_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """任务2：K-Means 聚类 + 手肘法 / 轮廓系数"""
    X = df[FEATURES].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    metrics = pd.DataFrame({"k": list(K_RANGE), "inertia": inertias, "silhouette": silhouettes})
    metrics.to_csv(OUT_DIR / "cluster_metrics.csv", index=False)

    # 手肘图 + 轮廓系数图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics["k"], metrics["inertia"], "bo-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].axvspan(4, 6, alpha=0.15, color="green", label="建议区间 4-6")
    axes[0].legend()

    axes[1].plot(metrics["k"], metrics["silhouette"], "ro-")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    for k in K_SUGGEST:
        axes[1].axvline(k, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_elbow_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 在 4-6 中选轮廓系数最高的 k
    sub = metrics[metrics["k"].isin(K_SUGGEST)]
    best_k = int(sub.loc[sub["silhouette"].idxmax(), "k"])

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = km_final.fit_predict(X_scaled)
    centers_orig = scaler.inverse_transform(km_final.cluster_centers_)

    # 聚类散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x=INCOME_COL, y=SPEND_COL, hue="Cluster", palette="tab10", s=80, alpha=0.85
    )
    plt.scatter(
        centers_orig[:, 0], centers_orig[:, 1], c="black", s=200, marker="X", label="Centroids"
    )
    plt.title(f"K-Means Clustering (k={best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_clusters.png", dpi=120, bbox_inches="tight")
    plt.close()

    df.to_csv(OUT_DIR / "Mall_Customers_clustered.csv", index=False)
    return df, best_k, metrics


def task3_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """任务3：分群画像统计"""
    profiles = []
    for c in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == c]
        male_pct = (sub["Gender"] == "Male").mean() * 100
        profiles.append(
            {
                "Cluster": int(c),
                "人数": len(sub),
                "平均年龄": round(sub["Age"].mean(), 1),
                "平均年收入_k": round(sub[INCOME_COL].mean(), 1),
                "平均消费评分": round(sub[SPEND_COL].mean(), 1),
                "男性占比_%": round(male_pct, 1),
                "女性占比_%": round(100 - male_pct, 1),
            }
        )
    prof_df = pd.DataFrame(profiles)
    prof_df.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)

    # 分组柱状图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.barplot(data=prof_df, x="Cluster", y="平均年龄", ax=axes[0], palette="Blues_d")
    sns.barplot(data=prof_df, x="Cluster", y="平均年收入_k", ax=axes[1], palette="Oranges_d")
    sns.barplot(data=prof_df, x="Cluster", y="平均消费评分", ax=axes[2], palette="Greens_d")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_cluster_profiles.png", dpi=120, bbox_inches="tight")
    plt.close()

    return prof_df


def _persona_key(inc: float, sp: float, income_med: float, spend_med: float) -> str:
    """按收入-消费四象限划分客群类型（相对全样本中位数）。"""
    high_inc, high_sp = inc > income_med, sp > spend_med
    if high_inc and high_sp:
        return "富裕积极型"
    if high_inc and not high_sp:
        return "富裕谨慎型"
    if not high_inc and high_sp:
        return "潜力活跃型"
    if not high_inc and not high_sp:
        return "普通节约型"
    return "中等稳健型"  # 接近中位数的边界客群


def cluster_labels_text(df: pd.DataFrame, prof_df: pd.DataFrame) -> dict[int, str]:
    """根据收入-消费四象限生成文字画像"""
    desc = {}
    income_med = df[INCOME_COL].median()
    spend_med = df[SPEND_COL].median()

    behavior_map = {
        "富裕积极型": "收入与消费评分均高于中位数，属于高价值核心客户，消费意愿强、忠诚度培养空间大。",
        "富裕谨慎型": "收入高但消费评分偏低，对价格或品牌较敏感，需通过高端体验与专属权益激活消费。",
        "潜力活跃型": "收入一般但消费意愿强，年轻冲动消费特征明显，适合促销、联名与会员积分驱动复购。",
        "普通节约型": "收入与消费评分均偏低，注重性价比，可通过基础款套装、满减与实用品类引流。",
        "中等稳健型": "收入与消费接近整体中等水平，消费习惯稳定，适合常规会员运营与品类组合推荐。",
    }

    for _, row in prof_df.iterrows():
        c = int(row["Cluster"])
        inc, sp = row["平均年收入_k"], row["平均消费评分"]
        persona = _persona_key(inc, sp, income_med, spend_med)
        desc[c] = (
            f"【客群 {c} — {persona}】共 {int(row['人数'])} 人。"
            f"平均年龄 {row['平均年龄']} 岁，平均年收入 {inc} k$，平均消费评分 {sp}；"
            f"男性占 {row['男性占比_%']}%，女性占 {row['女性占比_%']}%。"
            f"{behavior_map[persona]}"
        )
    return desc


def task4_marketing(df: pd.DataFrame, prof_df: pd.DataFrame) -> pd.DataFrame:
    """任务4：营销策略"""
    income_med = df[INCOME_COL].median()
    spend_med = df[SPEND_COL].median()

    marketing_db = {
        "富裕积极型": {
            "定位": "高净值高消费核心客群",
            "建议": [
                ("推出 VIP 私人导购与新品预览会", "该群消费力强，重视尊贵体验，专属服务可提升客单价与粘性。"),
                ("联合奢侈/轻奢品牌做限定联名", "高消费意愿匹配高端联名，能强化品牌认同并刺激冲动购买。"),
            ],
        },
        "富裕谨慎型": {
            "定位": "高收入低消费的理性客群",
            "建议": [
                ("发放高额满减券（满额可用）+ 保价承诺", "降低决策顾虑，用确定性优惠撬动高收入但低消费的沉睡需求。"),
                ("家庭套装 / 品质生活馆场景陈列", "强调实用与品质，契合理性消费心理，提高连带率。"),
            ],
        },
        "潜力活跃型": {
            "定位": "年轻高活跃、价格敏感型客群",
            "建议": [
                ("会员积分翻倍 + 快闪店/网红打卡活动", "消费评分高但收入有限，社交与积分更能驱动频繁到店。"),
                ("学生活动日、第二件半价等促销", "对折扣响应明显，短期促销可快速提升销量与客流。"),
            ],
        },
        "普通节约型": {
            "定位": "大众基础消费客群",
            "建议": [
                ("每周特价日 + 基础款组合包", "价格导向明显，明码低价与组合优惠最易转化。"),
                ("免费停车/公交接驳 + 社区便民服务", "提升到店便利性，以低成本权益培养习惯性到店。"),
            ],
        },
        "中等稳健型": {
            "定位": "中等收入、中等消费的稳定客群",
            "建议": [
                ("分级会员礼遇 + 生日月双倍积分", "消费习惯稳定，持续积分与等级权益可提升复购频率。"),
                ("家居、服饰等生活品类跨柜组合优惠", "客单价适中，组合促销可提高单次消费件数与停留时间。"),
            ],
        },
    }

    rows = []
    for _, row in prof_df.iterrows():
        c = int(row["Cluster"])
        key = _persona_key(
            row["平均年收入_k"], row["平均消费评分"], income_med, spend_med
        )
        m = marketing_db[key]
        rows.append(
            {
                "Cluster": c,
                "客群类型": key,
                "定位": m["定位"],
                "建议1": m["建议"][0][0],
                "理由1": m["建议"][0][1],
                "建议2": m["建议"][1][0],
                "理由2": m["建议"][1][1],
            }
        )

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(OUT_DIR / "marketing_strategies.csv", index=False)
    return strat_df


def main():
    print("=" * 50)
    print("Week2 商场客户聚类分析")
    print("=" * 50)

    df = load_data()
    print(f"\n[任务1] 数据规模: {df.shape}")
    print(f"缺失值:\n{df.isna().sum()}")

    info = task1_eda(df)
    print(f"IQR 异常值计数: {info['outliers_iqr']}")

    df, best_k, metrics = task2_clustering(df)
    print(f"\n[任务2] 建议区间 4-6 类轮廓系数: ")
    print(metrics[metrics["k"].isin(K_SUGGEST)][["k", "silhouette"]].to_string(index=False))
    print(f"选定 k = {best_k}")

    prof = task3_profiles(df)
    print(f"\n[任务3] 分群画像:\n{prof.to_string(index=False)}")

    desc = cluster_labels_text(df, prof)
    for c, text in sorted(desc.items()):
        print(f"\n{text}")

    task4_marketing(df, prof)
    print(f"\n结果已保存至 {FIG_DIR} 与 {OUT_DIR}")
    print("请结合 工程实训作业报告_week2.md 撰写/导出 PDF 提交。")


if __name__ == "__main__":
    main()
