"""
20 Newsgroups NLP 课后作业 — 完整可运行脚本

"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FIG = ROOT / "figures"

# 自选 4 个主题差异较大的类别
CATEGORIES_4 = [
    "comp.graphics",
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.misc",
]

STOP_WORDS = ENGLISH_STOP_WORDS
RANDOM_STATE = 42
TEST_SIZE = 0.3


def strip_headers_footers_quotes(text: str) -> str:
    """简易剔除邮件头、签名块、引用行（用于本地 txt 时的补充）。"""
    lines = []
    in_header = True
    for line in text.splitlines():
        s = line.strip()
        if in_header:
            if s == "":
                in_header = False
            continue
        if s.startswith(">") or s.startswith("|"):
            continue
        if re.match(r"^[-_=*]{3,}$", s):
            continue
        lines.append(line)
    body = "\n".join(lines)
    # 去掉常见页脚标记之后的内容
    for marker in ("\n-- \n", "\n---", "\nSent from"):
        if marker in body:
            body = body.split(marker)[0]
    return body.strip()


def preprocess_text(text: str, *, stem: bool = False, lemmatize: bool = False) -> str:
    """小写、去标点数字、分词、去停用词；可选词干/词形还原。"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]

    if stem or lemmatize:
        try:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass

    if stem:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    if lemmatize:
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def load_newsgroups(categories: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(data.data), data.target, list(data.target_names)


def vectorize_and_split(
    texts: list[str],
    y: np.ndarray,
    *,
    use_tfidf: bool = False,
    max_features: int = 20000,
):
    preprocessed = texts  # 已在外部预处理
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = Vectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        min_df=2,
    )
    X_train_v = vec.fit_transform(X_train)
    X_test_v = vec.transform(X_test)
    name = "TF-IDF" if use_tfidf else "Count (BoW)"
    return X_train_v, X_test_v, y_train, y_test, vec, name


def train_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    target_names: list[str],
    feature_name: str,
    model_name: str = "MultinomialNB",
):
    if model_name == "MultinomialNB":
        clf = MultinomialNB()
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    else:
        clf = LinearSVC(random_state=RANDOM_STATE, dual="auto")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    return acc, cm, report, y_pred


def plot_confusion(cm, labels, title: str, path: Path):
    plt.figure(figsize=(6, 5))
    short = [l.replace(".", "\n") for l in labels]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short, yticklabels=short)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def task1_load_and_eda():
    print("\n" + "=" * 60)
    print("【任务1】加载数据（剔除 headers / footers / quotes）")
    print("=" * 60)
    texts, y, names = load_newsgroups(CATEGORIES_4)
    print(f"选定类别: {names}")
    print(f"文档总数: {len(texts)}")
    print(f"类别数: {len(names)}")
    dist = pd.Series(y).value_counts().sort_index()
    dist.index = names
    print("\n各类样本数量分布:")
    print(dist.to_string())
    dist.to_csv(OUT / "task1_class_distribution.csv")
    return texts, y, names


def task2_preprocess(texts: list[str], *, stem=False, lemmatize=False) -> list[str]:
    tag = []
    if stem:
        tag.append("stem")
    if lemmatize:
        tag.append("lemma")
    label = "基础" if not tag else "+".join(tag)
    print(f"\n【任务2】文本预处理 ({label})")
    processed = [preprocess_text(t, stem=stem, lemmatize=lemmatize) for t in texts]
    print(f"示例原文前 120 字符:\n  {texts[0][:120]}...")
    print(f"示例预处理后:\n  {processed[0][:120]}...")
    return processed


def task3_4_nb_comparison(processed: list[str], y: np.ndarray, names: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("【任务3-4】特征表示 + 多项式朴素贝叶斯（7:3 划分）")
    print("=" * 60)
    rows = []
    for use_tfidf in (False, True):
        X_tr, X_te, y_tr, y_te, vec, feat = vectorize_and_split(
            processed, y, use_tfidf=use_tfidf
        )
        print(f"\n--- {feat} | 训练集 {X_tr.shape[0]} 条, 测试集 {X_te.shape[0]} 条, 特征维 {X_tr.shape[1]} ---")
        acc, cm, report, _ = train_evaluate(X_tr, X_te, y_tr, y_te, names, feat)
        print(f"测试集准确率: {acc:.4f}")
        print("混淆矩阵:")
        print(cm)
        print("\n分类报告:\n", report)
        slug = "bow" if not use_tfidf else "tfidf"
        plot_confusion(cm, names, f"NB + {feat}", FIG / f"cm_nb_{slug}_4class.png")
        rows.append({"特征": feat, "模型": "MultinomialNB", "准确率": round(acc, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task4_nb_comparison.csv", index=False)
    return df


def task5_analysis(nb_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("【任务5】词袋 vs TF-IDF 对比分析")
    print("=" * 60)
    bow = nb_df.loc[nb_df["特征"].str.contains("Count"), "准确率"].iloc[0]
    tfidf = nb_df.loc[nb_df["特征"].str.contains("TF"), "准确率"].iloc[0]
    diff = tfidf - bow
    analysis = f"""
词袋模型测试准确率: {bow:.4f}
TF-IDF 测试准确率:  {tfidf:.4f}
差值 (TF-IDF - 词袋): {diff:+.4f}

简要分析:
1. 两者均基于词频统计，适合多项式朴素贝叶斯的离散特征假设。
2. TF-IDF 降低高频公共词权重、突出区分度高的词，在主题分类中通常略优或相近。
3. 若 TF-IDF 更高：说明抑制"全文常见词"有助于区分 comp/sci/rec/talk 等主题。
4. 若词袋更高：可能因样本量不大，IDF 缩放反而削弱部分有判别力的高频词。
5. 本实验 4 类任务上二者差异一般不大，均能有效完成多类文本分类。
"""
    print(analysis)
    (OUT / "task5_analysis.txt").write_text(analysis.strip(), encoding="utf-8")


def optional_20_classes(processed_fn):
    print("\n" + "=" * 60)
    print("【选做1】全部 20 类多分类")
    print("=" * 60)
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    texts = processed_fn(list(data.data))
    y = data.target
    names = list(data.target_names)
    print(f"文档数: {len(texts)}, 类别数: {len(names)}")
    rows = []
    for use_tfidf in (False, True):
        X_tr, X_te, y_tr, y_te, _, feat = vectorize_and_split(texts, y, use_tfidf=use_tfidf)
        acc, cm, _, _ = train_evaluate(X_tr, X_te, y_tr, y_te, names, feat)
        print(f"{feat} + NB 准确率: {acc:.4f}")
        rows.append({"特征": feat, "类别数": 20, "准确率": round(acc, 4)})
    pd.DataFrame(rows).to_csv(OUT / "optional_20class.csv", index=False)


def optional_stem_lemma(texts, y, names):
    print("\n" + "=" * 60)
    print("【选做2】词干/词形还原对比")
    print("=" * 60)
    variants = [
        ("基础", False, False),
        ("词干还原", True, False),
        ("词形还原", False, True),
    ]
    rows = []
    for label, stem, lemma in variants:
        proc = task2_preprocess(texts, stem=stem, lemmatize=lemma)
        X_tr, X_te, y_tr, y_te, _, _ = vectorize_and_split(proc, y, use_tfidf=True)
        acc, _, _, _ = train_evaluate(X_tr, X_te, y_tr, y_te, names, "TF-IDF")
        print(f"  {label}: 准确率 {acc:.4f}")
        rows.append({"预处理": label, "TF-IDF+NB准确率": round(acc, 4)})
    pd.DataFrame(rows).to_csv(OUT / "optional_stem_lemma.csv", index=False)


def optional_model_compare(processed, y, names):
    print("\n" + "=" * 60)
    print("【选做3】LR / SVM / 朴素贝叶斯 对比（TF-IDF, 4类）")
    print("=" * 60)
    X_tr, X_te, y_tr, y_te, _, _ = vectorize_and_split(processed, y, use_tfidf=True)
    rows = []
    for model in ("MultinomialNB", "LogisticRegression", "LinearSVC"):
        acc, cm, report, _ = train_evaluate(
            X_tr, X_te, y_tr, y_te, names, "TF-IDF", model_name=model
        )
        print(f"\n{model}: 准确率 {acc:.4f}")
        print(report)
        rows.append({"模型": model, "特征": "TF-IDF", "准确率": round(acc, 4)})
        plot_confusion(
            cm, names, f"{model} + TF-IDF", FIG / f"cm_{model}_4class.png"
        )
    pd.DataFrame(rows).to_csv(OUT / "optional_model_compare.csv", index=False)


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    texts, y, names = task1_load_and_eda()
    processed = task2_preprocess(texts)
    nb_df = task3_4_nb_comparison(processed, y, names)
    task5_analysis(nb_df)

    optional_20_classes(lambda ts: [preprocess_text(t) for t in ts])
    optional_stem_lemma(texts, y, names)
    optional_model_compare(processed, y, names)

    print("\n全部结果已保存至:", OUT)
    print("混淆矩阵图:", FIG)


if __name__ == "__main__":
    main()
