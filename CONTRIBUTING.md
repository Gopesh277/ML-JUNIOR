# Contributing to ML-JUNIOR

Thank you for your interest in contributing to ML-JUNIOR! 

ML-JUNIOR is an open-source Python package designed to automate common machine-learning workflows such as dataset profiling, preprocessing, model selection, hyperparameter tuning, model saving, and experiment reporting.

We welcome bug reports, feature requests, documentation improvements, and code contributions.

## Ways to Contribute

You can contribute by:

* Reporting bugs
* Suggesting new features
* Improving documentation
* Adding tests
* Improving preprocessing
* Adding supported machine-learning models
* Improving model selection or tuning
* Improving experiment reports
* Fixing issues marked `good first issue`
* Reviewing pull requests

## Getting Started

### 1. Fork the repository

Fork the ML-JUNIOR repository on GitHub.

### 2. Clone your fork

```bash
git clone https://github.com/YOUR_USERNAME/ML-JUNIOR.git
cd ML-JUNIOR
```

### 3. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install the package locally

```bash
python -m pip install --upgrade pip
pip install -e .
```

### 5. Create a branch

Create a separate branch for your changes:

```bash
git checkout -b feature/my-feature
```

For a bug fix:

```bash
git checkout -b fix/my-bug
```

## Making Changes

Before submitting a pull request:

1. Keep changes focused.
2. Follow the existing project structure.
3. Write clear and readable Python code.
4. Update documentation when necessary.
5. Add or update tests when appropriate.
6. Make sure the existing workflow still works.

## Running the Demo

You can run the ML-JUNIOR demo with:

```bash
mljunior --demo --quick
```

The demo should complete successfully and generate the expected output files.

## Pull Requests

When opening a pull request, please explain:

* What you changed
* Why you changed it
* How you tested it
* Any limitations or known issues

A good pull request should have a clear title and description.

Example:

```text
Add feature importance to experiment report
```

## Issues

Before opening a new issue, check whether a similar issue already exists.

When reporting a bug, include:

* Python version
* ML-JUNIOR version
* Operating system
* Dataset type/format
* Command or code used
* Error message
* Steps to reproduce the problem

Please do not upload private or sensitive datasets.

## Good First Issues

If you are new to the project, look for issues labeled:

* `good first issue`
* `help wanted`
* `documentation`

These are good places to start contributing.

## Development Principles

ML-JUNIOR aims to remain:

* Simple to use
* Lightweight
* Reproducible
* Well documented
* Beginner friendly
* Useful for practical machine-learning workflows

Contributions should support these principles whenever possible.

## Code of Conduct

By participating in this project, you agree to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to ML-JUNIOR, you agree that your contributions will be licensed under the same license as the project.

Thank you for helping improve ML-JUNIOR! 
