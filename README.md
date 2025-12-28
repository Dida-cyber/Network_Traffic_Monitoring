# Network Traffic Monitoring - Machine Learning

Demo project showcasing data processing and H2O AutoML on a collection of CSV files (network/traffic). The main notebook is `pipelin-1.ipynb`.

## Contents
- `pipelin-1.ipynb`: full pipeline (CSV loading, cleaning, sampling, conversion to H2OFrame, AutoML, model saving)
- `MachineLearningCSV/`: folder containing the input CSV files
- `models/`: output folder where the best H2O model is saved

## Requirements
- Python 3.8+ (3.10 recommended)
- Java JRE/JDK (H2O requires Java; ensure Java is installed and `JAVA_HOME` is set)
- Main Python packages: `h2o`, `pandas`, `numpy`, `matplotlib`

Example install:

```bash
pip install h2o pandas numpy matplotlib
```

Note: if you have a `requirements.txt`, use `pip install -r requirements.txt`.

## How to run
1. Open `pipelin-1.ipynb` in Jupyter / VS Code (Jupyter extension) and run the cells in order.
2. Before running, ensure the `MachineLearningCSV` folder contains the CSV files (the notebook raises an error otherwise).
3. Adjust H2O memory if needed: `h2o.init(max_mem_size="5G")`. For better results, increase memory to 8G+ or run in the cloud if your machine lacks RAM.

## What I changed
- Fixed minor bugs: inconsistent use of `df` vs `df_sample` in missing-value checks and updated `dropna` so it modifies `df_sample` correctly.

## Results & interpretation
- The notebook shows the AutoML leaderboard and evaluates the best model on the test set.
- Key metrics to watch: LogLoss, MSE/RMSE, Mean Per-Class Error. If Mean Per-Class Error is high, inspect per-class metrics (precision/recall/F1) and the confusion matrix — rare classes are often poorly predicted.

## Data processing
- **Loading & concatenation**: the notebook finds all CSVs in `MachineLearningCSV`, reads each file with `pd.read_csv(..., sep=';')` (files use ';' as separator), and concatenates them into a single DataFrame `df`.
- **Column cleaning**: trims leading/trailing spaces from column names (`cols = cols.str.strip()`), normalizes names to avoid issues when converting to H2OFrame.
- **Sampling / imbalance handling**: creates a sample (`df_sample`) by grouping on the target and sampling with different fractions (e.g., `frac=0.20` for `BENIGN`, `0.80` for others) to reduce class imbalance before training.
- **Missing values**: inspects missing values using `isna().sum()`, shows percentage of missing values, and drops rows with missing `Flow Bytes/s` (`df_sample = df_sample.dropna(subset=['Flow Bytes/s'])`).
- **Memory optimization**: downcasts numeric types (`downcast='integer'`/`'float'`) to reduce memory usage when necessary.
- **Conversion to H2O**: after cleaning, `df_sample` is converted to `h2o.H2OFrame(df_sample)` and columns are trimmed again before train/test split.

## Methodology
- **Split**: train/test split with `hf.split_frame(ratios=[0.8], seed=1234)`.
- **AutoML (H2O)**: uses `H2OAutoML` with main parameters:
  - `max_runtime_secs=1800` (30 minutes search limit),
  - `balance_classes=True` (attempts to rebalance classes during training),
  - `stopping_metric='logloss'` (stopping criterion),
  - `project_name='Final'` and `seed=123` for reproducibility.
- **Evaluation**: retrieves the `leaderboard`, selects the `leader` model, and evaluates on the test set (`leader_model.model_performance(test)`) — shows LogLoss, MSE/RMSE, Mean Per-Class Error and confusion matrix.
- **Save**: final model is saved locally with `h2o.save_model(..., path='models')` for reuse or deployment.