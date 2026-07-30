"""Step 8: Generate a markdown experiment report summarizing the whole run."""

from datetime import datetime


def generate_report(
    profile: dict,
    cleanup_notes: list,
    task_type: str,
    suggestions: list,
    screen_scores: dict,
    tuned_results: list,
    skip_holdout: bool,
    best_name: str,
    best_params: dict,
    model_path: str,
    out_path: str,
):
    lines = []
    lines.append("# ML JUNIOR — experiment report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("## 1. Dataset")
    lines.append(f"- Rows: {profile['n_rows']}")
    lines.append(f"- Columns: {profile['n_cols']}")
    lines.append(f"- Target: `{profile['target']}`")
    lines.append(f"- Task type: **{task_type}**")
    if profile["duplicate_rows"]:
        lines.append(f"- Duplicate rows found: {profile['duplicate_rows']}")
    if profile["missing_cols"]:
        lines.append("- Missing values found:")
        for col, desc in profile["missing_cols"].items():
            lines.append(f"  - `{col}`: {desc}")
    else:
        lines.append("- Missing values: none detected")
    if profile["class_balance"]:
        lines.append("- Class balance:")
        for cls, pct in profile["class_balance"].items():
            lines.append(f"  - `{cls}`: {pct}%")
    lines.append("")

    lines.append("## 2. Cleaning applied")
    if cleanup_notes:
        for note in cleanup_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- Dataset was already clean, nothing to do")
    lines.append("")

    lines.append("## 3. Preprocessing plan")
    for s in suggestions:
        lines.append(f"- {s}")
    if not suggestions:
        lines.append("- No special preprocessing needed")
    lines.append("")

    lines.append("## 4. Quick screen (default hyperparameters, cross-validated on training data only)")
    lines.append("| Model | Score |")
    lines.append("|---|---|")
    for name, score in sorted(screen_scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        lines.append(f"| {name} | {score if score is not None else 'failed'} |")
    lines.append("")

    lines.append("## 5. Final leaderboard")
    if skip_holdout:
        lines.append(
            "_Dataset was too small for a reliable held-out test set, so these are "
            "cross-validated scores rather than true holdout performance._"
        )
        lines.append("")
        lines.append("| Rank | Model | CV score | Best hyperparameters |")
        lines.append("|---|---|---|---|")
        for i, row in enumerate(tuned_results, 1):
            params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in row["params"].items()) or "defaults"
            lines.append(f"| {i} | {row['name']} | {row['cv_score']} | {params_str} |")
    else:
        lines.append(
            "_These scores are measured on a held-out test set the models never saw during "
            "training or hyperparameter tuning — this is the honest estimate of real-world performance._"
        )
        lines.append("")
        metric_keys = list(tuned_results[0]["holdout_metrics"].keys())
        header = "| Rank | Model | " + " | ".join(metric_keys) + " | CV score (search estimate) | Best hyperparameters |"
        sep = "|---" * (len(metric_keys) + 4) + "|"
        lines.append(header)
        lines.append(sep)
        for i, row in enumerate(tuned_results, 1):
            metric_vals = " | ".join(str(row["holdout_metrics"][m]) for m in metric_keys)
            params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in row["params"].items()) or "defaults"
            lines.append(f"| {i} | {row['name']} | {metric_vals} | {row['cv_score']} | {params_str} |")
    lines.append("")

    lines.append("## 6. Best model")
    lines.append(f"**{best_name}**")
    if best_params:
        lines.append("")
        lines.append("Best hyperparameters:")
        for k, v in best_params.items():
            lines.append(f"- `{k.split('__')[-1]}`: {v}")
    lines.append("")
    lines.append(
        f"The saved model was refit on the **full cleaned dataset** (train + held-out test "
        f"combined) after the honest performance estimate above was measured — this maximizes "
        f"real-world performance without affecting the reported metrics."
    )
    lines.append("")
    lines.append(f"Saved to: `{model_path}`")
    lines.append("")

    lines.append("## 7. How to use the saved model")
    lines.append("```python")
    lines.append("import joblib")
    lines.append("")
    lines.append(f"saved = joblib.load('{model_path}')")
    lines.append("pipeline = saved['pipeline']")
    lines.append("label_encoder = saved['label_encoder']  # None for regression tasks")
    lines.append("feature_columns = saved['feature_columns']  # columns the model expects, in order")
    lines.append("")
    lines.append("# new_data_df: pandas DataFrame with these same feature columns")
    lines.append("# (drop any ID columns first — see saved['dropped_id_columns'])")
    lines.append("predictions = pipeline.predict(new_data_df[feature_columns])")
    lines.append("")
    lines.append("if label_encoder is not None:")
    lines.append("    predictions = label_encoder.inverse_transform(predictions)  # back to original labels")
    lines.append("```")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
