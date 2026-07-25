# ML JUNIOR

An agent that does the model-selection and hyperparameter-tuning grind for
you: point it at a dataset, and it reads it, figures out the problem type,
suggests and applies preprocessing, trains and tunes several algorithms,
saves the best one, and writes an experiment report — announcing every step
as it happens.

```
  __  __ _          _ _   _ _   _ ___ ___  ____  
|  \/  | |        | | | | | \ | |_ _/ _ \|  _ \ 
| |\/| | |        | | | | |  \| || | | | | |_) |
| |  | | |___     | | |_| | |\  || | |_| |  _ < 
|_|  |_|_____|    |_|\___/|_| \_|___\___/|_| \_\

   your ML training + tuning agent
```

## Install (one time)

```bash
unzip mljunior.zip
cd ml-junior
pip install -e .
```

`mljunior` is now a system command, usable from any folder — no `cd`-ing
back into this repo required.

## Use it

```bash
mljunior --demo                                     # try it on a built-in dataset
mljunior --data datasets/your_file.csv --target y   # your own dataset
mljunior                                            # no flags — it asks interactively
mljunior --data data.csv --target y --quick         # faster: fewer CV folds/tuning iters
```

Supports CSV, Excel (`.xlsx`), JSON, and Parquet. CSVs are read with automatic
encoding detection (handles Excel exports that aren't UTF-8).

## What it does — step by step

1. **Reads the dataset** — any of the formats above, profiles rows/columns/missingness/class balance
2. **Identifies the problem type** — classification or regression, from the target column
3. **Suggests + applies preprocessing** — imputation, one-hot/label encoding, scaling, class-imbalance handling — all explained in plain English before it's applied
4. **Trains multiple algorithms** — cross-validated quick screen with default hyperparameters
5. **Tunes hyperparameters** — `RandomizedSearchCV` on the top performers
6. **Compares metrics** — a final leaderboard (accuracy/F1 for classification — F1 automatically when classes are imbalanced; R2 for regression)
7. **Saves the best model** — `mljunior_output/best_model.joblib`, a full sklearn Pipeline (preprocessing + model together)
8. **Generates an experiment report** — `mljunior_output/experiment_report.md`, documenting every step above

## Using the saved model later

```python
import joblib

saved = joblib.load('mljunior_output/best_model.joblib')
pipeline = saved['pipeline']
label_encoder = saved['label_encoder']  # None for regression tasks

predictions = pipeline.predict(new_data_df)  # same feature columns as training data
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)  # back to original labels
```

## Adding your own dataset

Drop a CSV/Excel/JSON/Parquet file into the `datasets/` folder (or anywhere
else — it's just a suggested spot):

```bash
mljunior --data datasets/my_data.csv --target my_target_column
```

## Uninstall

```bash
pip uninstall mljunior
```
