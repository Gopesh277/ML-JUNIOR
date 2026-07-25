#!/usr/bin/env python3
"""
ML JUNIOR — an agent that does the hyperparameter-tuning grind for you.

Reads a dataset, figures out the problem type, suggests preprocessing,
trains and tunes several algorithms, saves the best one, and writes an
experiment report. Run `mljunior --demo` to see it end to end.
"""

import argparse
import os
import sys

import joblib
from sklearn.model_selection import train_test_split

from mljunior.data import load_data, load_demo_data, profile_dataset, print_profile
from mljunior.preprocessing import (
    detect_task_type, suggest_preprocessing, build_pipeline_preprocessor,
    preencode_high_cardinality, encode_target,
)
from mljunior.modeling import get_model_catalog, quick_screen, tune_model, scoring_metric
from mljunior.report import generate_report

BANNER = r"""
  __  __ _          _ _   _ _   _ ___ ___  ____  
|  \/  | |        | | | | | \ | |_ _/ _ \|  _ \ 
| |\/| | |        | | | | |  \| || | | | | |_) |
| |  | | |___     | | |_| | |\  || | |_| |  _ < 
|_|  |_|_____|    |_|\___/|_| \_|___\___/|_| \_\

   your ML training + tuning agent
"""


def step(n, title):
    print(f"\n[Step {n}] {title}")
    print("-" * 50)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="ML JUNIOR: read data, tune models, save the best one, write a report.")
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
    y_raw = df[target]
    X = df.drop(columns=[target])
    task_type = detect_task_type(y_raw)
    print(f"Detected: {task_type}")

    # --- Step 3: Suggest + apply preprocessing ---
    step(3, "Preprocessing plan")
    suggestions = suggest_preprocessing(X, y_raw, task_type)
    for s in suggestions:
        print(f"  - {s}")
    if not suggestions:
        print("  - No special preprocessing needed")

    preprocessor, id_cols, high_card_cols = build_pipeline_preprocessor(X)
    if id_cols:
        X = X.drop(columns=id_cols)
    X = preencode_high_cardinality(X, [c for c in high_card_cols if c in X.columns])
    y, label_encoder = encode_target(y_raw, task_type)

    imbalanced = False
    if task_type == "classification":
        counts = y_raw.value_counts(normalize=True)
        imbalanced = (counts.max() / max(counts.min(), 1e-9)) > 3

    # --- Step 4-5: Train multiple models, quick cross-val screen ---
    step(4, "Training candidate models (quick cross-validated screen)")
    cv_folds = 3
    catalog = get_model_catalog(task_type, class_weight_balanced=imbalanced)
    screen_scores = quick_screen(preprocessor, catalog, X, y, task_type, cv=cv_folds, imbalanced=imbalanced)
    metric_name = scoring_metric(task_type, imbalanced)
    if imbalanced:
        print(f"  (using {metric_name} instead of accuracy since classes are imbalanced)")
    for name, score in sorted(screen_scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        print(f"  {name:<22} {metric_name}={score}")

    # Shortlist top models for full tuning (skip ones that failed entirely)
    ranked = sorted(
        [(n, s) for n, s in screen_scores.items() if s is not None],
        key=lambda kv: -kv[1],
    )
    n_iter = 4 if args.quick else 8
    shortlist = [name for name, _ in ranked[: 4 if not args.quick else 2]]

    # --- Step 6: Hyperparameter tuning ---
    step(5, f"Tuning hyperparameters for top {len(shortlist)} models")
    tuned_results = []
    fitted_pipelines = {}
    for name in shortlist:
        estimator, param_dist = catalog[name]
        print(f"  Tuning {name}...")
        pipe, best_params, score, std = tune_model(
            preprocessor, estimator, param_dist, X, y, task_type, n_iter=n_iter, cv=cv_folds, imbalanced=imbalanced
        )
        tuned_results.append({"name": name, "score": score, "std": std, "params": best_params})
        fitted_pipelines[name] = pipe

    tuned_results.sort(key=lambda r: -r["score"])

    # --- Step 6b: Compare metrics ---
    step(6, "Final leaderboard")
    print(f"{'Rank':<5}{'Model':<22}{metric_name:<10}{'Std':<8}")
    for i, row in enumerate(tuned_results, 1):
        std_display = row["std"] if row["std"] is not None else "-"
        print(f"{i:<5}{row['name']:<22}{row['score']:<10}{std_display}")

    best = tuned_results[0]
    best_name = best["name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} ({metric_name}={best['score']})")

    # --- Step 7: Save best model ---
    step(7, "Saving the best model")
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "best_model.joblib")
    joblib.dump({"pipeline": best_pipeline, "label_encoder": label_encoder, "target": target}, model_path)
    print(f"Saved to: {model_path}")

    # --- Step 8: Generate experiment report ---
    step(8, "Writing experiment report")
    report_path = os.path.join(args.out_dir, "experiment_report.md")
    generate_report(
        profile=profile,
        task_type=task_type,
        suggestions=suggestions,
        screen_scores=screen_scores,
        tuned_results=tuned_results,
        best_name=best_name,
        best_params=best["params"],
        model_path=model_path,
        out_path=report_path,
    )
    print(f"Saved to: {report_path}")

    print("\nDone. ML JUNIOR finished its shift.")


if __name__ == "__main__":
    main()
