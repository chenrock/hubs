# Hubs

## Overview

This a data processing and modeling pipeline that analyzes freemium usage data to predict customer conversion within a configurable time window. It loads raw CSVs, transforms and engineers features, trains an XGBoost classifier, evaluates performance, and generates SHAP explanations.

## Project Structure

```
.
├── data
│   ├── raw                # source CSVs
│   │   ├── customers_(4).csv
│   │   ├── noncustomers_(4).csv
│   │   └── usage_actions_(4).csv
│   └── processed          # Parquet outputs
├── models                 # trained model & outputs
├── notebooks              # exploratory analysis
│   └── 00_eda_raw_data.ipynb
├── src
│   ├── data               # data loading & transformation
│   │   ├── features.py
│   │   └── transform.py
│   └── models             # training & prediction modules
│       ├── train.py
│       └── predict.py
├── .gitignore             # list for git to ignore
├── .python-version        # python version
├── main.py                # orchestrates pipeline
├── pyproject.toml         # project metadata & dependencies
├── README.md              # this file
└── uv.lock                # lockfile for dependencies
```

## Installation

Setup environment:

```bash
uv sync
```


## Data Description

### Raw Data (`data/raw`)

- **customers_(4).csv**: Sample list of paying customers.
- **noncustomers_(4).csv**: Companies not currently paying.
- **usage_actions_(4).csv**: Usage logs (freemium and paid activity).

### Column Definitions

| Column | Description |
|--------|-------------|
| `id` | Unique account identifier (portal id). |
| `WHEN_TIMESTAMP` | Timestamp of usage event. |
| `CLOSEDATE` | Date when an account converted to paying. |
| `MRR` | Monthly Recurring Revenue (float). |
| `ALEXA_RANK` | Traffic-based rank (1 best, 16,000,001 worst). |
| `EMPLOYEE_RANGE` | Company size bucket. |
| `INDUSTRY` | Raw industry string. |
| `ACTIONS_CRM_CONTACTS`, `ACTIONS_CRM_COMPANIES`, `ACTIONS_CRM_DEALS`, `ACTIONS_EMAIL` | Counts of object actions per timestamp. |
| `USERS_CRM_CONTACTS`, `USERS_CRM_COMPANIES`, `USERS_CRM_DEALS`, `USERS_EMAIL` | Number of users performing each action. |


## Pipeline

0. **Exploratory analysis**: `notebooks/00_eda_raw_data.ipynb`
1. **Data loading & cleaning**: `src/data/transform.py`
2. **Feature engineering & target creation**: `src/data/features.py`
3. **Model training & evaluation**: `src/models/train.py`
4. **Prediction**: `src/models/predict.py`

## Usage

### Run full pipeline

```bash
uv run main.py \
  --window 28 \
  --model-output models/xgb_28d.json \
  --validation-output data/processed/validation_28d.parquet \
  --feature-importance-output models/feature_importance_28d.csv \
  --shap-output models/shap_summary_28d.csv \
  --shap-plots-dir models/shap_plots \
  --threshold 0.2
```

- `--window`: conversion window in days (default: 28)
- `--skip-model`: skip training step
- `--skip-shap`: skip SHAP calculation

### Make predictions

```bash
python src/models/predict.py \
  --model models/xgb_28d.json \
  --data data/processed/data_with_target_28d.parquet \
  --output predictions.parquet \
  --threshold 0.2
```

### Exploratory Notebook

Launch Jupyter and open:

```
notebooks/00_eda_raw_data.ipynb
```

## Code Formatting

Apply Ruff:

```bash
uv run ruff format --line-length 79 .
```
&
```bash
uv run ruff check --fix -v \
  --select E,F,I \
  --extend-select W292 \
  --line-length 79 \
  .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow code style & run ruff --fix
