"""Step 3: Look at the data, suggest a preprocessing plan in plain English,
and build the actual sklearn ColumnTransformer pipeline that implements it."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

MAX_MISSING_FRACTION = 0.6  # drop a column if more than this fraction is missing


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes a categorical column as how common each value is (its
    frequency in the training data), instead of an arbitrary integer label.

    Why not LabelEncoder: assigning 0/1/2/3 to e.g. Chicago/Boston/NYC/LA
    invents a false ranking that linear models, SVMs, and KNN all interpret
    literally. Frequency encoding carries real signal (rare vs. common
    category) without inventing an order.

    Because this is a proper sklearn transformer, it only learns frequencies
    from whatever data it's fit on — inside cross-validation that means only
    the training fold, never the held-out fold. Unseen categories at
    transform time get frequency 0.
    """

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        self.freq_maps_ = {
            col: X[col].astype(str).value_counts(normalize=True) for col in X.columns
        }
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = np.zeros((len(X), len(self.columns_)))
        for i, col in enumerate(self.columns_):
            out[:, i] = X[col].astype(str).map(self.freq_maps_[col]).fillna(0.0).values
        return out

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _looks_like_id_column(col: pd.Series) -> bool:
    """True ID columns are either non-numeric all-unique values (names, UUIDs,
    emails) or a numeric sequential index (0,1,2,...). A continuous numeric
    feature like income or price can also be all-unique by chance — that's
    normal, predictive data, not an ID, so it must NOT be dropped."""
    if col.nunique(dropna=True) != len(col):
        return False
    if not pd.api.types.is_numeric_dtype(col):
        return True
    sorted_vals = col.sort_values().reset_index(drop=True)
    diffs = sorted_vals.diff().dropna()
    return bool((diffs == 1).all())


def detect_task_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= max(10, int(0.05 * len(y))):
        return "classification"
    return "regression"


def clean_dataset(df: pd.DataFrame, target: str) -> tuple:
    """Actually cleans the data (not just reports on it). Returns
    (cleaned_df, cleanup_notes: list[str])."""
    notes = []
    df = df.copy()

    n_before = len(df)
    missing_target = df[target].isna().sum()
    if missing_target:
        df = df[df[target].notna()]
        notes.append(f"Dropped {missing_target} row(s) with a missing target value")

    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
        notes.append(f"Dropped {dupes} duplicate row(s)")

    # Replace +/-inf (common after division-by-zero upstream) with NaN so the
    # imputer handles them like any other missing value.
    num_cols = df.select_dtypes(include="number").columns
    inf_count = np.isinf(df[num_cols]).sum().sum() if len(num_cols) else 0
    if inf_count:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        notes.append(f"Replaced {inf_count} infinite value(s) with missing (to be imputed)")

    X_cols = [c for c in df.columns if c != target]
    missing_frac = df[X_cols].isna().mean()
    high_missing = missing_frac[missing_frac > MAX_MISSING_FRACTION].index.tolist()
    if high_missing:
        df = df.drop(columns=high_missing)
        notes.append(
            f"Dropped {len(high_missing)} column(s) that are >{int(MAX_MISSING_FRACTION*100)}% "
            f"missing (too sparse to impute meaningfully): {high_missing}"
        )

    X_cols = [c for c in df.columns if c != target]
    constant_cols = [c for c in X_cols if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        notes.append(f"Dropped {len(constant_cols)} constant column(s) with no variation: {constant_cols}")

    if len(df) < n_before:
        notes.append(f"Rows: {n_before} -> {len(df)} after cleaning")

    return df, notes


def suggest_preprocessing(X: pd.DataFrame, y: pd.Series, task_type: str) -> list:
    """Returns a list of human-readable suggestion strings, and is also used
    to decide what the pipeline actually does."""
    suggestions = []

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    missing = X.isna().sum()
    if missing.any():
        suggestions.append(
            f"Impute remaining missing values (median for numeric, most-frequent for categorical) "
            f"in {(missing > 0).sum()} column(s)"
        )

    if cat_cols:
        high_card = [c for c in cat_cols if X[c].nunique() > 20]
        low_card = [c for c in cat_cols if X[c].nunique() <= 20]
        if low_card:
            suggestions.append(f"One-hot encode {len(low_card)} low-cardinality categorical column(s): {low_card}")
        if high_card:
            suggestions.append(
                f"Frequency-encode {len(high_card)} high-cardinality column(s) (too many unique "
                f"values for one-hot; frequency encoding avoids inventing a false ranking "
                f"the way label-encoding would): {high_card}"
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

    dropped = [c for c in X.columns if _looks_like_id_column(X[c])]
    if dropped:
        suggestions.append(
            f"Drop {len(dropped)} likely ID column(s) with all-unique values (no predictive signal): {dropped}"
        )

    return suggestions


def build_pipeline_preprocessor(X: pd.DataFrame):
    """Builds the actual ColumnTransformer used inside every model's Pipeline.
    Everything here — imputation, encoding, scaling — lives inside the
    pipeline, so it's refit fresh on every CV fold's training data only."""
    id_cols = [c for c in X.columns if _looks_like_id_column(X[c])]
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
        high_card_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("freq_encode", FrequencyEncoder()),
        ])
        transformers.append(("cat_high", high_card_pipeline, high_card))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, id_cols


def encode_target(y: pd.Series, task_type: str):
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le
    return y, None
