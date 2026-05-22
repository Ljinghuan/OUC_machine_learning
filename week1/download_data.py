"""Download Default of Credit Card Clients dataset (UCI / Kaggle mirror)."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
]


def download(output: Path | None = None) -> Path:
    output = output or Path(__file__).resolve().parent / "UCI_Credit_Card.csv"
    output = Path(output)

    print(f"Downloading from UCI archive...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    xls_name = next(n for n in archive.namelist() if n.lower().endswith(".xls"))
    with archive.open(xls_name) as f:
        df = pd.read_excel(f, header=1)

    df.columns = COLUMNS
    df.to_csv(output, index=False)
    print(f"Saved {output} ({len(df):,} rows, {len(df.columns)} columns)")
    return output


if __name__ == "__main__":
    download()
