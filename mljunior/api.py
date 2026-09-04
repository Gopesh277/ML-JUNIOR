import os
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from mljunior.data import (
    load_data,
    load_demo_data,
    profile_dataset,
)

from mljunior.preprocessing import (
    detect_task_type,
    clean_dataset,
    suggest_preprocessing,
    build_pipeline_preprocessor,
    encode_target,
)

from mljunior.modeling import (
    get_model_catalog,
    quick_screen,
    tune_model,
    scoring_metric,
    evaluate_on_holdout,
)

from mljunior.report import generate_report


MIN_ROWS_FOR_HOLDOUT = 20


class MLJunior:
    """
    Python API for ML-JUNIOR.

    Automatically:
    - loads data
    - cleans data
    - detects classification/regression
    - builds preprocessing
    - screens multiple models
    - tunes the best candidates
    - evaluates them
    - saves the best model
    - generates an experiment report
    """

    def __init__(
        self,
        data=None,
        target=None,
        out_dir="mljunior_output",
        quick=False,
        demo=False,
    ):
        self.data = data
        self.target = target
        self.out_dir = out_dir
        self.quick = quick
        self.demo = demo

        self.result = None

    def run(self):
        """Run the complete ML-JUNIOR pipeline."""

        # ---------------------------------------------------------
        # 1. Load dataset
        # ---------------------------------------------------------

        if self.demo:
            df, target = load_demo_data()
        elif self.data:
            df = load_data(self.data)

            if not self.target:
                raise ValueError(
                    "target is required when using a custom dataset."
                )

            target = self.target

        else:
            raise ValueError(
                "Provide data='path/to/data.csv' or use demo=True."
            )

        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # ---------------------------------------------------------
        # 2. Profile
        # ---------------------------------------------------------

        profile = profile_dataset(df, target)

        # ---------------------------------------------------------
        # 3. Detect task
        # ---------------------------------------------------------

        task_type = detect_task_type(df[target])

        # ---------------------------------------------------------
        # 4. Clean dataset
        # ---------------------------------------------------------

        df, cleanup_notes = clean_dataset(df, target)

        y_raw = df[target]
        X = df.drop(columns=[target])

        # ---------------------------------------------------------
        # 5. Preprocessing
        # ---------------------------------------------------------

        suggestions = suggest_preprocessing(
            X,
            y_raw,
            task_type
        )

        preprocessor, id_cols = build_pipeline_preprocessor(X)

        if id_cols:
            X = X.drop(columns=id_cols)

        y, label_encoder = encode_target(
            y_raw,
            task_type
        )

        # ---------------------------------------------------------
        # 6. Detect imbalance
        # ---------------------------------------------------------

        imbalanced = False

        if task_type == "classification":
            counts = y_raw.value_counts(normalize=True)

            imbalanced = (
                counts.max() /
                max(counts.min(), 1e-9)
            ) > 3

        # ---------------------------------------------------------
        # 7. Train/test split
        # ---------------------------------------------------------

        skip_holdout = len(X) < MIN_ROWS_FOR_HOLDOUT

        if (
            task_type == "classification"
            and not skip_holdout
        ):
            class_counts = pd.Series(y).value_counts()

            if class_counts.min() < 2:
                skip_holdout = True

        if skip_holdout:

            X_train = X
            y_train = y

            X_test = None
            y_test = None

        else:

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                    if task_type == "classification"
                    else None,
                )

            except ValueError:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                )

        # ---------------------------------------------------------
        # 8. Model screening
        # ---------------------------------------------------------

        cv_folds = 3

        catalog = get_model_catalog(
            task_type,
            class_weight_balanced=imbalanced
        )

        screen_scores = quick_screen(
            preprocessor,
            catalog,
            X_train,
            y_train,
            task_type,
            cv=cv_folds,
            imbalanced=imbalanced,
        )

        metric_name = scoring_metric(
            task_type,
            imbalanced
        )

        ranked = sorted(
            [
                (name, score)
                for name, score in screen_scores.items()
                if score is not None
            ],
            key=lambda item: -item[1]
        )

        if not ranked:
            raise RuntimeError(
                "Every candidate model failed during cross-validation."
            )

        # ---------------------------------------------------------
        # 9. Hyperparameter tuning
        # ---------------------------------------------------------

        n_iter = 4 if self.quick else 8

        shortlist = [
            name
            for name, _ in ranked[
                :2 if self.quick else 4
            ]
        ]

        tuned_results = []
        fitted_pipelines = {}

        for name in shortlist:

            estimator, param_dist = catalog[name]

            pipe, best_params, cv_score, std = tune_model(
                preprocessor,
                estimator,
                param_dist,
                X_train,
                y_train,
                task_type,
                n_iter=n_iter,
                cv=cv_folds,
                imbalanced=imbalanced,
            )

            row = {
                "name": name,
                "cv_score": cv_score,
                "std": std,
                "params": best_params,
            }

            if not skip_holdout:

                holdout_metrics = evaluate_on_holdout(
                    pipe,
                    X_test,
                    y_test,
                    task_type,
                )

                primary_key = (
                    "F1"
                    if task_type == "classification"
                    and imbalanced
                    else (
                        "Accuracy"
                        if task_type == "classification"
                        else "R2"
                    )
                )

                row["holdout_metrics"] = holdout_metrics
                row["rank_score"] = holdout_metrics[primary_key]

            else:

                row["holdout_metrics"] = None
                row["rank_score"] = cv_score

            tuned_results.append(row)

            fitted_pipelines[name] = pipe

        # ---------------------------------------------------------
        # 10. Select best model
        # ---------------------------------------------------------

        tuned_results.sort(
            key=lambda row: -row["rank_score"]
        )

        best = tuned_results[0]

        best_name = best["name"]

        best_pipeline = fitted_pipelines[
            best_name
        ]

        # ---------------------------------------------------------
        # 11. Refit on full dataset
        # ---------------------------------------------------------

        final_pipeline = clone(
            best_pipeline
        )

        final_pipeline.fit(
            X,
            y
        )

        # ---------------------------------------------------------
        # 12. Save model
        # ---------------------------------------------------------

        os.makedirs(
            self.out_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            self.out_dir,
            "best_model.joblib"
        )

        joblib.dump(
            {
                "pipeline": final_pipeline,
                "label_encoder": label_encoder,
                "target": target,
                "task_type": task_type,
                "feature_columns": list(X.columns),
                "dropped_id_columns": id_cols,
            },
            model_path,
        )

        # ---------------------------------------------------------
        # 13. Generate report
        # ---------------------------------------------------------

        report_path = os.path.join(
            self.out_dir,
            "experiment_report.md"
        )

        generate_report(
            profile=profile,
            cleanup_notes=cleanup_notes,
            task_type=task_type,
            suggestions=suggestions,
            screen_scores=screen_scores,
            tuned_results=tuned_results,
            skip_holdout=skip_holdout,
            best_name=best_name,
            best_params=best["params"],
            model_path=model_path,
            out_path=report_path,
        )

        # ---------------------------------------------------------
        # 14. Return result
        # ---------------------------------------------------------

        self.result = {
            "best_model": best_name,
            "task_type": task_type,
            "metric": metric_name,
            "score": best["rank_score"],
            "model_path": model_path,
            "report_path": report_path,
            "screen_scores": screen_scores,
            "tuned_results": tuned_results,
        }

        return self.result

    def predict(self, data):
        """
        Load the saved model and make predictions.

        data can be:
        - pandas DataFrame
        - path to CSV/XLSX/JSON/Parquet
        """

        model_path = os.path.join(
            self.out_dir,
            "best_model.joblib"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run agent.run() first."
            )

        saved = joblib.load(model_path)

        pipeline = saved["pipeline"]
        label_encoder = saved["label_encoder"]

        if isinstance(data, pd.DataFrame):
            df = data.copy()

        elif isinstance(data, str):
            df = load_data(data)

        else:
            raise TypeError(
                "data must be a pandas DataFrame "
                "or a dataset path."
            )

        predictions = pipeline.predict(df)

        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(
                predictions
            )

        return predictions
