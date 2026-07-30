"""Step 4-6: Train multiple models with cross-validation, then tune
hyperparameters on the top performers."""

import warnings

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def get_model_catalog(task_type: str, class_weight_balanced: bool = False):
    """Returns {name: (estimator, param_distributions)} for RandomizedSearchCV."""
    cw = "balanced" if class_weight_balanced else None

    if task_type == "classification":
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight=cw),
                {"model__C": [0.01, 0.1, 1, 10, 100]},
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=42, class_weight=cw),
                {"model__max_depth": [3, 5, 10, None], "model__min_samples_leaf": [1, 2, 5]},
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42, class_weight=cw),
                {"model__n_estimators": [100, 200, 400], "model__max_depth": [None, 10, 20],
                 "model__min_samples_leaf": [1, 2, 4]},
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=42),
                {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1, 0.2],
                 "model__max_depth": [2, 3, 4]},
            ),
            "SVM": (
                SVC(probability=True, random_state=42, class_weight=cw),
                {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
            ),
            "K-Nearest Neighbors": (
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 5, 7, 9, 15]},
            ),
            "Naive Bayes": (
                GaussianNB(),
                {},
            ),
        }
    else:
        return {
            "Linear Regression": (LinearRegression(), {}),
            "Decision Tree": (
                DecisionTreeRegressor(random_state=42),
                {"model__max_depth": [3, 5, 10, None], "model__min_samples_leaf": [1, 2, 5]},
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {"model__n_estimators": [100, 200, 400], "model__max_depth": [None, 10, 20]},
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(random_state=42),
                {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1, 0.2],
                 "model__max_depth": [2, 3, 4]},
            ),
            "SVR": (
                SVR(),
                {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
            ),
            "K-Nearest Neighbors": (
                KNeighborsRegressor(),
                {"model__n_neighbors": [3, 5, 7, 9, 15]},
            ),
        }


def scoring_metric(task_type: str, imbalanced: bool = False) -> str:
    if task_type == "classification":
        return "f1_weighted" if imbalanced else "accuracy"
    return "r2"


def quick_screen(preprocessor, catalog, X, y, task_type, cv=3, imbalanced=False):
    """Step 4-5: quick cross-val score for every model with default params,
    to shortlist which ones deserve full hyperparameter tuning."""
    scoring = scoring_metric(task_type, imbalanced)
    scores = {}
    for name, (estimator, _) in catalog.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            scores[name] = round(float(np.mean(cv_scores)), 4)
        except Exception:
            scores[name] = None
    return scores


def tune_model(preprocessor, estimator, param_dist, X, y, task_type, n_iter=8, cv=3, imbalanced=False):
    """Step 6: hyperparameter tuning via RandomizedSearchCV for one model."""
    scoring = scoring_metric(task_type, imbalanced)
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])

    if not param_dist:
        # No hyperparameters to tune (e.g. Naive Bayes, Linear Regression) —
        # just cross-validate for a stable score, then fit on all data.
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X, y)
        return pipe, {}, round(float(np.mean(cv_scores)), 4), round(float(np.std(cv_scores)), 4)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=min(n_iter, _grid_size(param_dist)),
        cv=cv, scoring=scoring, random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, round(search.best_score_, 4), None


def _grid_size(param_dist: dict) -> int:
    size = 1
    for v in param_dist.values():
        size *= len(v)
    return size


def evaluate_on_holdout(pipeline, X_test, y_test, task_type: str) -> dict:
    """The honest number: how the tuned pipeline performs on data it never
    saw during training OR hyperparameter search. This — not the CV score
    from the search — is what should be reported as the model's accuracy."""
    preds = pipeline.predict(X_test)
    if task_type == "classification":
        return {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
        }
    else:
        return {
            "R2": round(r2_score(y_test, preds), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        }
