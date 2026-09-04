# ML JUNIOR

> **Your ML training + tuning agent.**

ML JUNIOR is a Python machine-learning automation library that takes a dataset, automatically analyzes the problem, applies preprocessing, trains and tunes multiple machine-learning models, selects the best-performing model, saves it, and generates an experiment report.

You can use ML JUNIOR in **Python code** or directly from the **command line**.

```text
  __  __ _          _ _   _ _   _ ___ ___  ____  
|  \/  | |        | | | | | \ | |_ _/ _ \|  _ \ 
| |\/| | |        | | | | |  \| || | | | | |_) |
| |  | | |___     | | |_| | |\  || | |_| |  _ <
|_|  |_|_____|    |_|\___/|_| \_|___\___/|_| \_\

        your ML training + tuning agent
```

---

## Installation

Install ML JUNIOR directly from PyPI:

```bash
pip install mljunior
```

Verify the installation:

```bash
mljunior --help
```

You can also verify the Python API:

```bash
python -c "from mljunior import MLJunior; print(MLJunior)"
```

---

# Python API

The recommended way to use ML JUNIOR inside Python applications is through the `MLJunior` class.

## Basic usage

```python
from mljunior import MLJunior

agent = MLJunior(
    data="data.csv",
    target="target"
)

result = agent.run()

print("Best model:", result["best_model"])
print("Task type:", result["task_type"])
print("Score:", result["score"])
```

ML JUNIOR will automatically:

```text
Dataset
   ↓
Data loading
   ↓
Data cleaning
   ↓
Problem detection
   ↓
Preprocessing
   ↓
Model screening
   ↓
Hyperparameter tuning
   ↓
Model evaluation
   ↓
Best model
   ↓
Saved model + experiment report
```

---

## Quick mode

For faster experimentation, use `quick=True`:

```python
from mljunior import MLJunior

agent = MLJunior(
    data="data.csv",
    target="target",
    quick=True
)

result = agent.run()

print(result)
```

Quick mode uses fewer cross-validation folds and fewer hyperparameter-search iterations.

Use quick mode when experimenting with datasets or developing your application.

For a more thorough search, leave `quick=False`.

---

# Built-in Demo

You can test ML JUNIOR without providing your own dataset:

```python
from mljunior import MLJunior

agent = MLJunior(
    demo=True,
    quick=True
)

result = agent.run()

print(result)
```

This is useful for quickly verifying that the installation and ML pipeline are working.

---

# Output Directory

By default, ML JUNIOR creates:

```text
mljunior_output/
├── best_model.joblib
└── experiment_report.md
```

You can choose another output directory:

```python
from mljunior import MLJunior

agent = MLJunior(
    data="data.csv",
    target="target",
    out_dir="my_results"
)

result = agent.run()
```

The resulting structure will be:

```text
my_results/
├── best_model.joblib
└── experiment_report.md
```

---

# Making Predictions

After training, ML JUNIOR can use the saved model to make predictions.

```python
import pandas as pd
from mljunior import MLJunior

agent = MLJunior(
    data="data.csv",
    target="target",
    out_dir="mljunior_output"
)

agent.run()

new_data = pd.DataFrame({
    "feature_1": [10],
    "feature_2": [25],
    "feature_3": [100]
})

predictions = agent.predict(new_data)

print(predictions)
```

The saved model contains the preprocessing pipeline and trained model together, so new data goes through the same preprocessing used during training.

---

# Using a Previously Trained Model

You can also load the saved model directly with `joblib`.

```python
import joblib

saved = joblib.load(
    "mljunior_output/best_model.joblib"
)

pipeline = saved["pipeline"]

predictions = pipeline.predict(new_data_df)

print(predictions)
```

For classification tasks, the saved object also contains the label encoder:

```python
label_encoder = saved["label_encoder"]

if label_encoder is not None:
    predictions = label_encoder.inverse_transform(
        predictions
    )
```

---

# Viewing Feature Names

The trained pipeline stores the feature names used during training.

```python
import joblib

saved = joblib.load(
    "mljunior_output/best_model.joblib"
)

pipeline = saved["pipeline"]

print(list(pipeline.feature_names_in_))
```

This is useful when preparing new data for prediction.

---

# Result Object

`agent.run()` returns a dictionary containing information about the experiment.

Example:

```python
result = agent.run()

print(result["best_model"])
print(result["task_type"])
print(result["score"])
print(result["model_path"])
print(result["report_path"])
```

The result includes information such as:

```text
best_model
task_type
score
metric
model_path
report_path
screen_scores
tuned_results
```

---

# Command-Line Interface

ML JUNIOR also provides a CLI for users who don't need to integrate it into Python code.

## Demo

```bash
mljunior --demo
```

For a faster demo:

```bash
mljunior --demo --quick
```

## Train on your dataset

```bash
mljunior --data datasets/your_file.csv --target target
```

Example:

```bash
mljunior --data datasets/housing.csv --target price
```

## Quick mode

```bash
mljunior \
    --data datasets/housing.csv \
    --target price \
    --quick
```

## Custom output directory

```bash
mljunior \
    --data datasets/housing.csv \
    --target price \
    --out-dir results
```

## Interactive mode

You can also run:

```bash
mljunior
```

and provide the required information interactively.

---

# Supported Dataset Formats

ML JUNIOR supports:

* CSV
* Excel (`.xlsx`)
* JSON
* Parquet

Example:

```python
from mljunior import MLJunior

agent = MLJunior(
    data="dataset.xlsx",
    target="price"
)

result = agent.run()
```

or:

```bash
mljunior --data dataset.xlsx --target price
```

CSV files are handled with automatic encoding detection to support common Excel-generated CSV files that may not use UTF-8.

---

# What ML JUNIOR Does

## 1. Data Loading & Profiling

ML JUNIOR reads the dataset and analyzes:

* Number of rows
* Number of columns
* Data types
* Missing values
* Target distribution
* Class balance

---

## 2. Problem Detection

ML JUNIOR automatically determines whether the task is:

```text
Classification
```

or:

```text
Regression
```

based on the target column.

---

## 3. Data Cleaning

The pipeline handles common data-quality issues and prepares the dataset for machine-learning algorithms.

---

## 4. Preprocessing

ML JUNIOR can automatically apply preprocessing such as:

* Missing-value imputation
* Categorical encoding
* Numerical scaling
* Feature handling
* Class-imbalance handling

The preprocessing is incorporated into the machine-learning pipeline.

---

## 5. Model Screening

Multiple machine-learning algorithms are evaluated using cross-validation.

This provides an initial comparison before expensive hyperparameter tuning.

---

## 6. Hyperparameter Tuning

The strongest candidate models are further optimized using:

```text
RandomizedSearchCV
```

This allows ML JUNIOR to search different hyperparameter combinations automatically.

---

## 7. Model Evaluation

Models are compared using appropriate metrics.

For classification:

* Accuracy
* F1 score when classes are imbalanced

For regression:

* R²

The final leaderboard identifies the strongest model.

---

## 8. Best Model

The selected model is saved as:

```text
best_model.joblib
```

The saved object contains the preprocessing pipeline and trained model together.

This means you don't have to manually reproduce preprocessing when making predictions.

---

## 9. Experiment Report

ML JUNIOR generates:

```text
experiment_report.md
```

The report documents the experiment, including the data analysis, preprocessing, models, tuning, evaluation and selected model.

---

# Example Project

A simple ML JUNIOR project can look like:

```text
my-ml-project/
│
├── data/
│   └── dataset.csv
│
├── train.py
│
└── requirements.txt
```

`train.py`:

```python
from mljunior import MLJunior

agent = MLJunior(
    data="data/dataset.csv",
    target="target",
    out_dir="results"
)

result = agent.run()

print("\nBest model:", result["best_model"])
print("Task:", result["task_type"])
print("Score:", result["score"])
print("Model saved to:", result["model_path"])
print("Report saved to:", result["report_path"])
```

Run:

```bash
python train.py
```

---

# Why Use ML JUNIOR?

Without ML JUNIOR, a typical experiment may require manually:

```text
Load dataset
    ↓
Clean data
    ↓
Analyze target
    ↓
Identify task
    ↓
Build preprocessing
    ↓
Select models
    ↓
Train models
    ↓
Cross-validation
    ↓
Hyperparameter tuning
    ↓
Evaluate
    ↓
Save model
    ↓
Document experiment
```

ML JUNIOR automates this workflow:

```text
                ML JUNIOR
                    │
                    ▼
                Dataset
                    │
                    ▼
             Automatic analysis
                    │
                    ▼
              Preprocessing
                    │
                    ▼
             Model selection
                    │
                    ▼
           Hyperparameter tuning
                    │
                    ▼
              Model evaluation
                    │
                    ▼
             ┌──────┴──────┐
             ▼             ▼
       Best Model       Report
             │
             ▼
          Predict
```

---

# API Reference

## `MLJunior`

```python
MLJunior(
    data=None,
    target=None,
    out_dir="mljunior_output",
    quick=False,
    demo=False
)
```

### Parameters

| Parameter | Type   | Description                                 |
| --------- | ------ | ------------------------------------------- |
| `data`    | `str`  | Path to CSV, Excel, JSON or Parquet dataset |
| `target`  | `str`  | Target/label column                         |
| `out_dir` | `str`  | Directory for model and report              |
| `quick`   | `bool` | Use a faster training/tuning configuration  |
| `demo`    | `bool` | Run using the built-in demo dataset         |

### Methods

#### `run()`

Runs the complete ML training and tuning pipeline.

```python
result = agent.run()
```

#### `predict(data)`

Makes predictions using the trained/saved model.

```python
predictions = agent.predict(new_data)
```

`data` can be a pandas DataFrame or a supported dataset path.

---

# Uninstall

```bash
pip uninstall mljunior
```

---

# License

ML JUNIOR is released under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) for the full license text.

---

# Author

**Gopesh**

GitHub: [Gopesh277/ML-JUNIOR](https://github.com/Gopesh277/ML-JUNIOR)

---

## Contributing

Contributions, bug reports and feature suggestions are welcome.

Before submitting a pull request:

1. Test the change locally.
2. Add or update tests where appropriate.
3. Update the documentation if the public API changes.
4. Keep the CLI and Python API behavior consistent.

---

## Project Status

Current release:

```text
ML JUNIOR 0.1.1
```

ML JUNIOR is actively being developed. Future versions may introduce additional models, preprocessing strategies, evaluation metrics, experiment tracking and deployment-oriented functionality.
