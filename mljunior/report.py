"""Step 8: Generate a markdown experiment report summarizing the whole run."""

from datetime import datetime


def generate_report(
    profile: dict,
    task_type: str,
    suggestions: list,
    screen_scores: dict,
    tuned_results: list,
    best_name: str,
    best_params: dict,
    model_path: str,
    out_path: str,
):
    lines = []
    lines.append(f"# ML JUNIOR — experiment report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("## 1. Dataset")
    lines.append(f"- Rows: {profile['n_rows']}")
    lines.append(f"- Columns: {profile['n_cols']}")
    lines.append(f"- Target: `{profile['target']}`")
    lines.append(f"- Task type: **{task_type}**")
    if profile["duplicate_rows"]:
        lines.append(f"- Duplicate rows: {profile['duplicate_rows']}")
    if profile["missing_cols"]:
        lines.append("- Missing values:")
        for col, desc in profile["missing_cols"].items():
            lines.append(f"  - `{col}`: {desc}")
    else:
        lines.append("- Missing values: none detected")
    if profile["class_balance"]:
        lines.append("- Class balance:")
        for cls, pct in profile["class_balance"].items():
            lines.append(f"  - `{cls}`: {pct}%")
    lines.append("")

    lines.append("## 2. Preprocessing applied")
    for s in suggestions:
        lines.append(f"- {s}")
    if not suggestions:
        lines.append("- No special preprocessing needed")
    lines.append("")

    lines.append("## 3. Quick screen (default hyperparameters, cross-validated)")
    lines.append("| Model | Score |")
    lines.append("|---|---|")
    for name, score in sorted(screen_scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        lines.append(f"| {name} | {score if score is not None else 'failed'} |")
    lines.append("")

    lines.append("## 4. Hyperparameter-tuned leaderboard")
    lines.append("| Rank | Model | Score | Std | Best hyperparameters |")
    lines.append("|---|---|---|---|---|")
    for i, row in enumerate(tuned_results, 1):
        params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in row["params"].items()) or "defaults"
        std_str = row["std"] if row["std"] is not None else "-"
        lines.append(f"| {i} | {row['name']} | {row['score']} | {std_str} | {params_str} |")
    lines.append("")

    lines.append("## 5. Best model")
    lines.append(f"**{best_name}**")
    if best_params:
        lines.append("")
        lines.append("Best hyperparameters:")
        for k, v in best_params.items():
            lines.append(f"- `{k.split('__')[-1]}`: {v}")
    lines.append("")
    lines.append(f"Saved to: `{model_path}`")
    lines.append("")

    lines.append("## 6. How to use the saved model")
    lines.append("```python")
    lines.append("import joblib")
    lines.append("")
    lines.append(f"saved = joblib.load('{model_path}')")
    lines.append("pipeline = saved['pipeline']")
    lines.append("label_encoder = saved['label_encoder']  # None for regression tasks")
    lines.append("")
    lines.append("# new_data_df: pandas DataFrame with the same feature columns used in training")
    lines.append("# (everything except the target column)")
    lines.append("predictions = pipeline.predict(new_data_df)")
    lines.append("")
    lines.append("if label_encoder is not None:")
    lines.append("    predictions = label_encoder.inverse_transform(predictions)  # back to original labels")
    lines.append("```")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
