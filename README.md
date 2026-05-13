# Workforce Attrition Analysis

Comprehensive people analytics project for a mid-to-large Philippine corporation, combining Decision Tree Classification, Hierarchical Clustering, and Regularized Regression to surface actionable HR insights from 15 years of employee data.

## Overview

This project cleans and analyzes 5,025 employee records spanning HR, payroll, and performance data, then applies three data mining techniques to predict attrition, segment the workforce, and benchmark compensation.

## Project Structure

```text
.
├── data.csv                  # Raw employee dataset (5,025 records)
├── question_01.py            # Data quality assessment and cleaning
├── question_02.py            # Descriptive statistics
├── question_03.py            # Correlation analysis
├── question_04.py            # Attrition frequency analysis
├── question_05.py            # Decision Tree classifier
├── question_06.py            # Model evaluation (confusion matrix, classification report)
├── question_07.py            # Feature importance
├── question_08.py            # Tree pruning and bias–variance tradeoff
├── question_09.py            # Decision path interpretation
├── question_10.py            # Data standardization (StandardScaler)
├── question_11.py            # Agglomerative Hierarchical Clustering
├── question_12.py            # Linkage method comparison (Average vs Complete)
├── question_13.py            # Cluster profiling
├── question_14.py            # Cluster and attrition analysis
├── question_15.py            # OLS Regression
├── question_16.py            # Multicollinearity and VIF analysis
├── question_17.py            # Ridge Regression
├── question_18.py            # Lasso Regression
├── question_19.py            # Elastic Net Regression
├── question_20.py            # Regression model comparison
├── question_21.py            # High-risk employee analysis
├── question_22.py            # Contradiction analysis
├── question_23.py            # CHRO recommendation report
├── bonus_01.py               # Decision Tree optimization
├── bonus_02.py               # Clustering comparison analysis
└── bonus_03.py               # Feature engineering and model improvement
```

## Setup

**Requirements:** Python ≥ 3.14

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Running the Analysis

Each module is independently executable. Run any question directly:

```bash
uv run src/question_01.py   # Data cleaning — always run this first
uv run src/question_05.py   # Decision Tree
uv run src/question_11.py   # Hierarchical Clustering
uv run src/question_15.py   # OLS Regression
uv run src/question_23.py   # Final CHRO report
```

Modules share cached artifacts via `@cache`-decorated loader functions (e.g., `get_clean_df()`, `get_model_artifacts()`), so running a downstream module also runs its dependencies automatically.

## Data Cleaning (`question_01.py`)

The raw dataset contains missing values, typographical inconsistencies, encoding errors, and impossible values. The cleaning pipeline:

- **Drops** rows with impossible ages (< 15 or > 80) and invalid salaries (≤ 0 or > ₱500,000)
- **Clips** negative absences and tenure to 0
- **Normalizes** department name variants (e.g., `"OPERATIONS"`, `"Ops"` → `"Operations"`)
- **Imputes** numeric columns with their median and categorical columns with their mode

## Analyses

### Decision Tree Classification

Target variable: `Attrition` (binary: stayed / left)

- Full unpruned tree trained with `random_state=42`, 70/30 stratified split
- Evaluated via precision, recall, F1, and confusion matrix
- Bias–variance tradeoff curve across all depths to identify the recommended pruning depth (depth 4)
- Feature importances ranked for the top 10 predictors
- Decision path traced for a hypothetical employee profile

### Hierarchical Clustering

Features: salary, performance, satisfaction, work-life balance, tenure, absences, training hours, overtime (all standardized via `StandardScaler`)

- 300-employee stratified sample, 3 clusters
- Average and Complete linkage compared; Complete linkage selected for better cluster balance
- Clusters named and profiled: **Stable Performers**, **Outlier Pair**, **Flight Risk**
- Box plots generated for all 8 features across clusters
- HR interventions recommended per cluster

### Regularized Regression

Target variable: `Monthly_Salary_PHP`

All four models use the same train/test split and feature set:

| Model | Purpose |
|---|---|
| OLS | Baseline; all features retained |
| Ridge (λ tuned via CV) | Shrinks coefficients, handles multicollinearity |
| Lasso (λ tuned via CV) | Sparse solution; eliminates irrelevant features |
| Elastic Net (λ + L1 ratio tuned via CV) | Combines Ridge and Lasso penalties |

VIF analysis (`question_16.py`) identifies multicollinearity between `Tenure_Years` and `Num_Promotions`; a `Promotion_Rate` engineered feature is used as a remedy.

### High-Risk Employee Identification (`question_21.py`)

Employees flagged by **both** the Decision Tree (predicted to leave) and the clustering model (Flight Risk cluster) are the highest-confidence departure risks. A salary gap analysis using the Lasso benchmark identifies those who are also underpaid relative to their predicted compensation.

### CHRO Recommendation (`question_23.py`)

Synthesizes all three models into an executive summary with:

- Retention priority tiers
- Salary fairness findings for the Flight Risk cluster
- Contradiction cases (DT predicts stay, but cluster = Flight Risk — "silent flight risks")
- Modeling limitations and caveats

## Key Findings

- The Decision Tree achieves strong test accuracy with overtime hours, job satisfaction, and monthly salary ranking among the top predictors of attrition.
- The Flight Risk cluster exhibits the highest attrition rate and the largest proportion of underpaid employees relative to the Lasso salary benchmark.
- Lasso and Elastic Net select a sparse subset of features for salary prediction, with regularized models generalizing comparably to OLS while reducing overfitting.
- Feature engineering (salary-per-tenure, overtime-to-satisfaction ratio, promotion rate, absence-to-tenure ratio) provides modest improvement in tree performance and interpretability.
