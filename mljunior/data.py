"""Step 1-2: Load the dataset and profile it (shape, dtypes, missingness, balance)."""

import os
import sys

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV, Excel, JSON, or Parquet. Tries multiple encodings for CSV
    since real-world files (especially Excel exports) are often not UTF-8."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".parquet":
        return pd.read_parquet(path)

    # CSV / TSV / anything else: try encodings, sniff delimiter
    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as e:
                last_error = e
                continue

    print(f"Error: could not read '{path}' with any common encoding ({last_error}).")
    sys.exit(1)


def load_demo_data():
    from sklearn.datasets import load_breast_cancer
    df = load_breast_cancer(as_frame=True).frame
    return df, "target"


def profile_dataset(df: pd.DataFrame, target: str) -> dict:
    """Build a summary dict describing the dataset — used both for console
    output and for the experiment report."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / n_rows * 100).round(1)
    missing_cols = {
        col: f"{missing[col]} missing ({missing_pct[col]}%)"
        for col in df.columns
        if missing[col] > 0
    }

    y = df[target]
    dtypes = df.drop(columns=[target]).dtypes.astype(str).to_dict()

    class_balance = None
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= max(10, int(0.05 * n_rows)):
        counts = y.value_counts(normalize=True).round(3) * 100
        class_balance = counts.to_dict()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target": target,
        "dtypes": dtypes,
        "missing_cols": missing_cols,
        "class_balance": class_balance,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def print_profile(profile: dict):
    print(f"Rows: {profile['n_rows']}   Columns: {profile['n_cols']}   Target: {profile['target']}")
    if profile["duplicate_rows"]:
        print(f"  Note: {profile['duplicate_rows']} duplicate rows found")
    if profile["missing_cols"]:
        print("  Missing values found in:")
        for col, desc in profile["missing_cols"].items():
            print(f"    - {col}: {desc}")
    else:
        print("  No missing values detected")
    if profile["class_balance"]:
        print("  Class balance:")
        for cls, pct in profile["class_balance"].items():
            print(f"    - {cls}: {pct}%")
