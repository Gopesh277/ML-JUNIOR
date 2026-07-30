#!/usr/bin/env python3
"""
ML JUNIOR — an agent that does the hyperparameter-tuning grind for you.

Reads a dataset, cleans it, figures out the problem type, suggests
preprocessing, trains and tunes several algorithms, evaluates the winner on
a genuine held-out test set, saves the best model, and writes an experiment
report. Run `mljunior --demo` to see it end to end.
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from mljunior.data import load_data, load_demo_data, profile_dataset, print_profile
from mljunior.preprocessing import (
    detect_task_type, clean_dataset, suggest_preprocessing,
    build_pipeline_preprocessor, encode_target,
)
from mljunior.modeling import (
    get_model_catalog, quick_screen, tune_model, scoring_metric, evaluate_on_holdout,
)
from mljunior.report import generate_report

BANNER = r"""
  __  __ _          _ _   _ _   _ ___ ___  ____  
|  \/  | |        | | | | | \ | |_ _/ _ \|  _ \ 
| |\/| | |        | | | | |  \| || | | | | |_) |
| |  | | |___     | | |_| | |\  || | |_| |  _ < 
|_|  |_|_____|    |_|\___/|_| \_|___\___/|_| \_\

   your ML training + tuning agent
"""

MIN_ROWS_FOR_HOLDOUT = 20


def step(n, title):
    print(f"\n[Step {n}] {title}")
    print("-" * 50)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="ML JUNIOR: clean data, tune models, save the best one, write a report.")
    parser.add_argument("--data", type=str, help="Path to dataset (csv, xlsx, json, or parquet)")
    parser.add_argument("--target", type=str, help="Target/label column name")
    parser.add_argument("--demo", action="store_true", help="Run on a built-in demo dataset")
    parser.add_argument("--out-dir", type=str, default="mljunior_output", help="Output directory for model + report")
    parser.add_argument("--quick", action="store_true", help="Faster run: fewer CV folds and tuning iterations")
    args = parser.parse_args()

    # --- Step 1: Read the dataset ---
    step(1, "Reading the dataset")
    if args.demo:
        df, target = load_demo_data()
        print("Using built-in demo dataset (breast cancer classification)")
    elif args.data and args.target:
        df = load_data(args.data)
        target = args.target
    elif args.data and not args.target:
        df = load_data(args.data)
        print(f"Columns found: {list(df.columns)}")
        target = input("Which column should I predict? ").strip()
    else:
        print("No dataset given.")
        path = input("Dataset path (or Enter for demo): ").strip()
        if not path:
            df, target = load_demo_data()
        else:
            df = load_data(path)
            print(f"Columns found: {list(df.columns)}")
            target = input("Which column should I predict? ").strip()

    if target not in df.columns:
        print(f"Error: target column '{target}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    profile = profile_dataset(df, target)
    print_profile(profile)

    # --- Step 2: Identify classification/regression ---
    step(2, "Identifying the problem type")
    task_type = detect_task_type(df[target])
    print(f"Detected: {task_type}")

    # --- Step 3: Clean the data, then suggest + apply preprocessing ---
    step(3, "Cleaning the dataset")
    df, cleanup_notes = clean_dataset(df, target)
    if cleanup_notes:
        for note in cleanup_notes:
            print(f"  - {note}")
    else:
        print("  - Dataset was already clean, nothing to do")

    y_raw = df[target]
    X = df.drop(columns=[target])

    print("\nPreprocessing plan:")
    suggestions = suggest_preprocessing(X, y_raw, task_type)
    for s in suggestions:
        print(f"  - {s}")
    if not suggestions:
        print("  - No special preprocessing needed")

    preprocessor, id_cols = build_pipeline_preprocessor(X)
    if id_cols:
        X = X.drop(columns=id_cols)
    y, label_encoder = encode_target(y_raw, task_type)

    imbalanced = False
    if task_type == "classification":
        counts = y_raw.value_counts(normalize=True)
        imbalanced = (counts.max() / max(counts.min(), 1e-9)) > 3

    # --- Held-out split: tuning never sees this data ---
    skip_holdout = len(X) < MIN_ROWS_FOR_HOLDOUT
    if task_type == "classification" and not skip_holdout:
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            skip_holdout = True

    if skip_holdout:
        print(f"\n(Dataset too small for a reliable held-out test set — reporting "
              f"cross-validated scores instead of true holdout performance)")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 4-5: Train multiple models, quick cross-val screen (on train only) ---
    step(4, "Training candidate models (quick cross-validated screen)")
    cv_folds = 3
    catalog = get_model_catalog(task_type, class_weight_balanced=imbalanced)
    screen_scores = quick_screen(preprocessor, catalog, X_train, y_train, task_type, cv=cv_folds, imbalanced=imbalanced)
    metric_name = scoring_metric(task_type, imbalanced)
    if imbalanced:
        print(f"  (using {metric_name} instead of accuracy since classes are imbalanced)")
    for name, score in sorted(screen_scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        print(f"  {name:<22} {metric_name}={score}")

    ranked = sorted(
        [(n, s) for n, s in screen_scores.items() if s is not None],
        key=lambda kv: -kv[1],
    )
    if not ranked:
        print("\nError: every candidate model failed during cross-validation.")
        print("This usually means the dataset is too small or too imbalanced for the "
              "requested number of CV folds (e.g. fewer than 2 samples in some class).")
        print("Try a larger dataset, or a target column with more examples per class.")
        sys.exit(1)

    n_iter = 4 if args.quick else 8
    shortlist = [name for name, _ in ranked[: 4 if not args.quick else 2]]

    # --- Step 5: Hyperparameter tuning (on train only) ---
    step(5, f"Tuning hyperparameters for top {len(shortlist)} models")
    tuned_results = []
    fitted_pipelines = {}
    for name in shortlist:
        estimator, param_dist = catalog[name]
        print(f"  Tuning {name}...")
        pipe, best_params, cv_score, std = tune_model(
            preprocessor, estimator, param_dist, X_train, y_train, task_type,
            n_iter=n_iter, cv=cv_folds, imbalanced=imbalanced,
        )
        row = {"name": name, "cv_score": cv_score, "std": std, "params": best_params}

        if not skip_holdout:
            holdout_metrics = evaluate_on_holdout(pipe, X_test, y_test, task_type)
            primary_key = "F1" if (task_type == "classification" and imbalanced) else (
                "Accuracy" if task_type == "classification" else "R2"
            )
            row["holdout_metrics"] = holdout_metrics
            row["rank_score"] = holdout_metrics[primary_key]
        else:
            row["holdout_metrics"] = None
            row["rank_score"] = cv_score

        tuned_results.append(row)
        fitted_pipelines[name] = pipe

    tuned_results.sort(key=lambda r: -r["rank_score"])

    # --- Step 6: Compare metrics ---
    step(6, "Final leaderboard")
    if skip_holdout:
        print(f"{'Rank':<5}{'Model':<22}{'CV ' + metric_name:<16}")
        for i, row in enumerate(tuned_results, 1):
            print(f"{i:<5}{row['name']:<22}{row['cv_score']:<16}")
    else:
        print("(Scores below are on a held-out test set the models never trained or tuned on)")
        header_metrics = list(tuned_results[0]["holdout_metrics"].keys())
        print(f"{'Rank':<5}{'Model':<22}" + "".join(f"{m:<12}" for m in header_metrics) + f"{'CV ' + metric_name:<14}")
        for i, row in enumerate(tuned_results, 1):
            metric_str = "".join(f"{row['holdout_metrics'][m]:<12}" for m in header_metrics)
            print(f"{i:<5}{row['name']:<22}{metric_str}{row['cv_score']:<14}")

    best = tuned_results[0]
    best_name = best["name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} ({'holdout' if not skip_holdout else 'CV'} {metric_name}={best['rank_score']})")

    # --- Step 7: Save best model (refit on ALL data for the production artifact) ---
    step(7, "Saving the best model")
    final_pipeline = clone(best_pipeline)
    final_pipeline.fit(X, y)  # refit on train+test combined, now that we've honestly measured generalization

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "best_model.joblib")
    joblib.dump({
        "pipeline": final_pipeline,
        "label_encoder": label_encoder,
        "target": target,
        "task_type": task_type,
        "feature_columns": list(X.columns),
        "dropped_id_columns": id_cols,
    }, model_path)
    print(f"Saved to: {model_path}")
    print("(Refit on the full cleaned dataset for production use, after honestly measuring "
          "generalization on the held-out split above)")

    # --- Step 8: Generate experiment report ---
    step(8, "Writing experiment report")
    report_path = os.path.join(args.out_dir, "experiment_report.md")
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
    print(f"Saved to: {report_path}")

    print("\nDone. ML JUNIOR finished its shift.")


if __name__ == "__main__":
    main()
