"""
信用卡违约预测 — 可执行分析脚本（与 notebook 逻辑一致）。
运行前请确保 week1/UCI_Credit_Card.csv 存在，或执行 download_data.py。
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

WEEK1_DIR = Path(__file__).resolve().parent
DATA_PATH = WEEK1_DIR / "UCI_Credit_Card.csv"
FIG_DIR = WEEK1_DIR / "figures"
TARGET = "default.payment.next.month"


def ensure_data() -> Path:
    if not DATA_PATH.exists():
        from download_data import download

        download(DATA_PATH)
    return DATA_PATH


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    data = df.copy()
    if "ID" in data.columns:
        data = data.drop(columns=["ID"])
    data["EDUCATION"] = data["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    data["MARRIAGE"] = data["MARRIAGE"].replace({0: 3})
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return X, y


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_pipeline(model, preprocess, sampler=None):
    steps = [("preprocess", preprocess)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


def plot_eda(y: pd.Series, limit_bal: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=y, ax=axes[0])
    axes[0].set_title("Target Distribution")
    sns.histplot(limit_bal, bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Credit Limit Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_eda.png", dpi=120)
    plt.close()


def plot_correlation(data: pd.DataFrame) -> None:
    corr = data.corr(numeric_only=True)
    top = corr[TARGET].abs().sort_values(ascending=False).head(12).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[top].corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Top related features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlation.png", dpi=120)
    plt.close()


def cross_validate_models(X_train, y_train, preprocess) -> pd.DataFrame:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    samplers = {
        "SMOTE": SMOTE(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        for sampler_name, sampler in [("None", None), *samplers.items()]:
            pipe = build_pipeline(model, preprocess, sampler)
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )
            results.append(
                {
                    "model": name,
                    "sampler": sampler_name.replace("RandomUnderSampler", "Under"),
                    **{k: np.mean(scores[f"test_{k}"]) for k in scoring},
                }
            )
            print(f"  {name} + {sampler_name}: F1={results[-1]['f1']:.4f}")

    return pd.DataFrame(results).sort_values(["f1", "roc_auc"], ascending=False)


def evaluate_best(
    X_train, X_test, y_train, y_test, preprocess, results_df: pd.DataFrame
):
    best = results_df.iloc[0]
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    samplers = {
        "SMOTE": SMOTE(random_state=42),
        "Under": RandomUnderSampler(random_state=42),
    }
    sampler = samplers.get(best["sampler"])
    pipe = build_pipeline(models[best["model"]], preprocess, sampler)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\n=== Best on test set ===")
    print(f"Model: {best['model']} | Sampler: {best['sampler']}")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_confusion_matrix.png", dpi=120)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].plot(
        recall,
        precision,
        label=f"PR-AUC = {average_precision_score(y_test, y_prob):.4f}",
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_roc_pr_curves.png", dpi=120)
    plt.close()

    return best, pipe


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    ensure_data()
    X, y = load_and_clean(DATA_PATH)
    data = pd.concat([X, y], axis=1)

    print("Dataset:", X.shape)
    print("Target distribution:\n", y.value_counts(normalize=True).round(4))

    plot_eda(y, X["LIMIT_BAL"])
    plot_correlation(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocess = build_preprocess(X)

    print("\nCross-validation (5-fold)...")
    results_df = cross_validate_models(X_train, y_train, preprocess)
    out_csv = WEEK1_DIR / "model_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison table -> {out_csv}")
    print(results_df.head(6).to_string(index=False))

    evaluate_best(X_train, X_test, y_train, y_test, preprocess, results_df)
    print(f"\nFigures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
