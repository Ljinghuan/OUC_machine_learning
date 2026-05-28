"""W5 二手车价格预测 — 脚本版（与 Notebook 流程一致，用于快速验证）"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "car.csv"
FIG = ROOT / "figures"
OUT = ROOT / "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CURRENT_YEAR = 2024


def parse_price(s: str) -> float:
    """统一为 Lakh（十万卢比）单位。"""
    s = str(s).strip()
    if "Crore" in s:
        return float(s.replace("Crore", "").strip()) * 100
    if "Lakh" in s:
        return float(s.replace("Lakh", "").strip())
    # 纯数字视为卢比，换算为 Lakh
    return float(s.replace(",", "")) / 100_000


def parse_kms(s: str) -> float:
    return float(str(s).replace(" kms", "").replace(",", "").strip())


def parse_engine(s: str) -> float:
    return float(str(s).replace(" cc", "").strip())


def parse_seats(s: str) -> int:
    return int(str(s).replace(" Seats", "").strip())


def extract_brand(name: str) -> str:
    return str(name).split()[0]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col=0)
    df = df.copy()
    df["price_lakh"] = df["car_prices_in_rupee"].map(parse_price)
    df["kms"] = df["kms_driven"].map(parse_kms)
    df["engine_cc"] = df["engine"].map(parse_engine)
    df["seats_num"] = df["Seats"].map(parse_seats)
    df["brand"] = df["car_name"].map(extract_brand)
    df["car_age"] = CURRENT_YEAR - df["manufacture"]
    df = df[df["price_lakh"] > 0]
    df = df[df["kms"] >= 0]
    return df


def evaluate(y_true, y_pred, name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"模型": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE%": mape}


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    df = load_and_clean()
    features = ["brand", "kms", "fuel_type", "transmission", "ownership", "manufacture", "engine_cc", "seats_num", "car_age"]
    X = df[features]
    y = df["price_lakh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cat_cols = ["brand", "fuel_type", "transmission", "ownership"]
    num_cols = ["kms", "manufacture", "engine_cc", "seats_num", "car_age"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    mlp = Pipeline(
        [
            ("pre", pre),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=400, random_state=RANDOM_STATE, early_stopping=True)),
        ]
    )
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)

    rows = [evaluate(y_test, pred_rf, "随机森林(传统ML)"), evaluate(y_test, pred_mlp, "MLP神经网络")]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_metrics.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
