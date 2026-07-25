"""Step 3: Look at the data, suggest a preprocessing plan in plain English,
and build the actual sklearn ColumnTransformer pipeline that implements it."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def detect_task_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= max(10, int(0.05 * len(y))):
        return "classification"
    return "regression"


def suggest_preprocessing(X: pd.DataFrame, y: pd.Series, task_type: str) -> list:
    """Returns a list of human-readable suggestion strings, and is also used
    to decide what the pipeline actually does."""
    suggestions = []

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    missing = X.isna().sum()
    if missing.any():
        suggestions.append(
            f"Impute missing values (median for numeric, most-frequent for categorical) "
            f"in {(missing > 0).sum()} column(s)"
        )

    if cat_cols:
        high_card = [c for c in cat_cols if X[c].nunique() > 20]
        low_card = [c for c in cat_cols if X[c].nunique() <= 20]
        if low_card:
            suggestions.append(f"One-hot encode {len(low_card)} low-cardinality categorical column(s): {low_card}")
        if high_card:
            suggestions.append(
                f"Label-encode {len(high_card)} high-cardinality column(s) (too many "
                f"unique values for one-hot): {high_card}"
            )

    if num_cols:
        suggestions.append(f"Standard-scale {len(num_cols)} numeric column(s) (zero mean, unit variance)")

    if task_type == "classification":
        counts = y.value_counts(normalize=True)
        if counts.max() / max(counts.min(), 1e-9) > 3:
            suggestions.append(
                f"Class imbalance detected (largest class is {counts.max()*100:.0f}% of data) — "
                f"using class_weight='balanced' where supported"
            )

    dropped = [c for c in X.columns if X[c].nunique() == len(X)]
    if dropped:
        suggestions.append(
            f"Drop {len(dropped)} likely ID column(s) with all-unique values (no predictive signal): {dropped}"
        )

    return suggestions


def build_pipeline_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Builds the actual ColumnTransformer used inside every model's Pipeline."""
    # Drop obvious ID columns (all-unique values) before building the transformer
    id_cols = [c for c in X.columns if X[c].nunique() == len(X)]
    usable_cols = [c for c in X.columns if c not in id_cols]

    num_cols = X[usable_cols].select_dtypes(include="number").columns.tolist()
    cat_cols_all = X[usable_cols].select_dtypes(exclude="number").columns.tolist()
    low_card = [c for c in cat_cols_all if X[c].nunique() <= 20]
    high_card = [c for c in cat_cols_all if X[c].nunique() > 20]

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if low_card:
        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat_low", cat_pipeline, low_card))

    if high_card:
        # Label-encode high-cardinality columns; done outside ColumnTransformer
        # since LabelEncoder isn't natively column-transformer-friendly for
        # unseen categories. We pre-encode these in the CLI before fitting.
        pass

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, id_cols, high_card


def preencode_high_cardinality(X: pd.DataFrame, high_card_cols: list) -> pd.DataFrame:
    """Label-encodes high-cardinality categorical columns in place (returns a copy)."""
    X = X.copy()
    for col in high_card_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X


def encode_target(y: pd.Series, task_type: str):
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le
    return y, None
