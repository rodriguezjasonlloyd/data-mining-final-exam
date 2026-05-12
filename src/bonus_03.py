"""Bonus 3. Feature Engineering and Model Improvement."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from question_01 import console, get_clean_df
from question_05 import DROP_COLUMNS, encode
from question_17 import RidgeArtifacts, get_ridge_artifacts
from question_18 import LassoArtifacts, get_lasso_artifacts

ENGINEERED_FEATURES: list[str] = [
    "salary_per_tenure",
    "overtime_to_satisfaction_ratio",
    "promotion_rate",
    "absence_to_tenure_ratio",
]

RANDOM_STATE = 42


class TreeComparison(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    baseline_accuracy: float
    baseline_f1: float
    engineered_accuracy: float
    engineered_f1: float


class RegressionComparison(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model_name: str
    baseline_test_r2: float
    baseline_test_rmse: float
    baseline_nonzero: int
    engineered_test_r2: float
    engineered_test_rmse: float
    engineered_nonzero: int


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["salary_per_tenure"] = result["Monthly_Salary_PHP"] / (result["Tenure_Years"] + 1)
    result["overtime_to_satisfaction_ratio"] = result["Overtime_Hours_Monthly"] / (result["Job_Satisfaction_Score"] + 1)
    result["promotion_rate"] = result["Num_Promotions"] / (result["Tenure_Years"] + 1)
    result["absence_to_tenure_ratio"] = result["Absences_YTD"] / (result["Tenure_Years"] + 1)
    return result


def prepare_tree(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = encode(df)
    x = encoded.drop(columns=DROP_COLUMNS)
    y = encoded["Attrition"]
    return x, y


def prepare_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    regression_drop = ["Employee_ID", "Hire_Date", "Monthly_Salary_PHP", "Attrition"]
    encoded = encode(df)
    x = encoded.drop(columns=regression_drop)
    y = df["Monthly_Salary_PHP"]
    return x, y


def evaluate_tree(x: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    y_prediction: np.ndarray = model.predict(x_test)
    accuracy: float = accuracy_score(y_test, y_prediction)
    f1: float = f1_score(y_test, y_prediction)
    return accuracy, f1


def evaluate_ols(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[float, float, int]:
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_prediction: np.ndarray = model.predict(x_test)
    test_r2: float = r2_score(y_test, y_prediction)
    test_rmse: float = float(np.sqrt(mean_squared_error(y_test, y_prediction)))
    nonzero: int = int((pd.Series(model.coef_) != 0).sum())
    return test_r2, test_rmse, nonzero


def evaluate_ridge(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, alpha: float) -> tuple[float, float, int]:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_prediction: np.ndarray = model.predict(x_test)
    test_r2: float = r2_score(y_test, y_prediction)
    test_rmse: float = float(np.sqrt(mean_squared_error(y_test, y_prediction)))
    nonzero: int = int((pd.Series(model.coef_) != 0).sum())
    return test_r2, test_rmse, nonzero


def evaluate_lasso(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, alpha: float) -> tuple[float, float, int]:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(x_train, y_train)
    y_prediction: np.ndarray = model.predict(x_test)
    test_r2: float = r2_score(y_test, y_prediction)
    test_rmse: float = float(np.sqrt(mean_squared_error(y_test, y_prediction)))
    nonzero: int = int((pd.Series(model.coef_) != 0).sum())
    return test_r2, test_rmse, nonzero


def compare_trees(df_baseline: pd.DataFrame, df_engineered: pd.DataFrame) -> TreeComparison:
    x_baseline, y_baseline = prepare_tree(df_baseline)
    x_engineered, y_engineered = prepare_tree(df_engineered)

    baseline_accuracy, baseline_f1 = evaluate_tree(x_baseline, y_baseline)
    engineered_accuracy, engineered_f1 = evaluate_tree(x_engineered, y_engineered)

    return TreeComparison(
        baseline_accuracy=baseline_accuracy,
        baseline_f1=baseline_f1,
        engineered_accuracy=engineered_accuracy,
        engineered_f1=engineered_f1,
    )


def compare_regression(
    df_baseline: pd.DataFrame,
    df_engineered: pd.DataFrame,
    ridge_artifacts: RidgeArtifacts,
    lasso_artifacts: LassoArtifacts,
) -> list[RegressionComparison]:
    x_baseline, y_baseline = prepare_regression(df_baseline)
    x_engineered, y_engineered = prepare_regression(df_engineered)

    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_baseline, y_baseline, test_size=0.3, random_state=RANDOM_STATE)
    x_train_e, x_test_e, y_train_e, y_test_e = train_test_split(x_engineered, y_engineered, test_size=0.3, random_state=RANDOM_STATE)

    ols_b = evaluate_ols(x_train_b, x_test_b, y_train_b, y_test_b)
    ols_e = evaluate_ols(x_train_e, x_test_e, y_train_e, y_test_e)

    ridge_b = evaluate_ridge(x_train_b, x_test_b, y_train_b, y_test_b, ridge_artifacts.optimal_alpha)
    ridge_e = evaluate_ridge(x_train_e, x_test_e, y_train_e, y_test_e, ridge_artifacts.optimal_alpha)

    lasso_b = evaluate_lasso(x_train_b, x_test_b, y_train_b, y_test_b, lasso_artifacts.optimal_alpha)
    lasso_e = evaluate_lasso(x_train_e, x_test_e, y_train_e, y_test_e, lasso_artifacts.optimal_alpha)

    return [
        RegressionComparison(
            model_name="OLS",
            baseline_test_r2=ols_b[0],
            baseline_test_rmse=ols_b[1],
            baseline_nonzero=ols_b[2],
            engineered_test_r2=ols_e[0],
            engineered_test_rmse=ols_e[1],
            engineered_nonzero=ols_e[2],
        ),
        RegressionComparison(
            model_name="Ridge",
            baseline_test_r2=ridge_b[0],
            baseline_test_rmse=ridge_b[1],
            baseline_nonzero=ridge_b[2],
            engineered_test_r2=ridge_e[0],
            engineered_test_rmse=ridge_e[1],
            engineered_nonzero=ridge_e[2],
        ),
        RegressionComparison(
            model_name="Lasso",
            baseline_test_r2=lasso_b[0],
            baseline_test_rmse=lasso_b[1],
            baseline_nonzero=lasso_b[2],
            engineered_test_r2=lasso_e[0],
            engineered_test_rmse=lasso_e[1],
            engineered_nonzero=lasso_e[2],
        ),
    ]


def report_engineered_features() -> None:
    console.print(
        Panel(
            "[bold]salary_per_tenure[/bold] = Monthly_Salary_PHP / (Tenure_Years + 1)\n"
            "  Measures compensation relative to loyalty. A low ratio flags long-tenured employees who may be underpaid compared to newer hires.\n\n"
            "[bold]overtime_to_satisfaction_ratio[/bold] = Overtime_Hours_Monthly / (Job_Satisfaction_Score + 1)\n"
            "  Captures workload-to-satisfaction imbalance. High values indicate employees working excessive overtime despite low satisfaction — a strong flight risk signal.\n\n"
            "[bold]promotion_rate[/bold] = Num_Promotions / (Tenure_Years + 1)\n"
            "  Measures career velocity. Employees with low promotion rates relative to tenure may feel stagnated, correlating with disengagement and attrition.\n\n"
            "[bold]absence_to_tenure_ratio[/bold] = Absences_YTD / (Tenure_Years + 1)\n"
            "  Normalizes absenteeism by tenure. High ratios in newer employees may indicate early disengagement; high ratios in senior employees may signal burnout.",
            title="Bonus 3 — Engineered Features: HR Interpretation",
        ),
    )


def report_tree_comparison(comparison: TreeComparison) -> None:
    table = Table(title="Decision Tree — Before vs After Feature Engineering", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("With Engineered Features", justify="right")
    table.add_column("Δ Change", justify="right")

    accuracy_delta: float = comparison.engineered_accuracy - comparison.baseline_accuracy
    f1_delta: float = comparison.engineered_f1 - comparison.baseline_f1

    accuracy_color = "green" if accuracy_delta >= 0 else "red"
    f1_color = "green" if f1_delta >= 0 else "red"

    table.add_row(
        "Test Accuracy",
        f"{comparison.baseline_accuracy:.4f}",
        f"{comparison.engineered_accuracy:.4f}",
        f"[{accuracy_color}]{accuracy_delta:+.4f}[/{accuracy_color}]",
    )
    table.add_row(
        "Test F1 Score",
        f"{comparison.baseline_f1:.4f}",
        f"{comparison.engineered_f1:.4f}",
        f"[{f1_color}]{f1_delta:+.4f}[/{f1_color}]",
    )

    console.print(table)


def report_regression_comparison(comparisons: list[RegressionComparison]) -> None:
    table = Table(title="Regression — Before vs After Feature Engineering", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Baseline R²", justify="right")
    table.add_column("Engineered R²", justify="right")
    table.add_column("Δ R²", justify="right")
    table.add_column("Baseline RMSE", justify="right")
    table.add_column("Engineered RMSE", justify="right")
    table.add_column("Δ RMSE", justify="right")
    table.add_column("Baseline NZ", justify="right")
    table.add_column("Engineered NZ", justify="right")

    for comp in comparisons:
        r2_delta: float = comp.engineered_test_r2 - comp.baseline_test_r2
        rmse_delta: float = comp.engineered_test_rmse - comp.baseline_test_rmse
        r2_color = "green" if r2_delta >= 0 else "red"
        rmse_color = "green" if rmse_delta <= 0 else "red"

        table.add_row(
            comp.model_name,
            f"{comp.baseline_test_r2:.4f}",
            f"{comp.engineered_test_r2:.4f}",
            f"[{r2_color}]{r2_delta:+.4f}[/{r2_color}]",
            f"{comp.baseline_test_rmse:,.2f}",
            f"{comp.engineered_test_rmse:,.2f}",
            f"[{rmse_color}]{rmse_delta:+,.2f}[/{rmse_color}]",
            str(comp.baseline_nonzero),
            str(comp.engineered_nonzero),
        )

    console.print(table)
    console.print("[dim]NZ = number of non-zero coefficients. Δ RMSE: negative is improvement (lower error).[/dim]")


def plot_correlation(df: pd.DataFrame) -> None:
    correlation_columns: list[str] = [*ENGINEERED_FEATURES, "Attrition", "Monthly_Salary_PHP"]
    correlation_matrix = df[correlation_columns].corr()

    _fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Correlation — Engineered Features vs Attrition and Salary")
    plt.tight_layout()
    plt.show()


def plot_tree_comparison(comparison: TreeComparison) -> None:
    categories: list[str] = ["Accuracy", "F1 Score"]
    baseline_values: list[float] = [comparison.baseline_accuracy, comparison.baseline_f1]
    engineered_values: list[float] = [comparison.engineered_accuracy, comparison.engineered_f1]

    x = np.arange(len(categories))
    width: float = 0.35

    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_values, width, label="Baseline", color="steelblue")
    ax.bar(x + width / 2, engineered_values, width, label="With Engineered Features", color="tomato")

    ax.set_title("Decision Tree — Accuracy and F1 Before vs After Feature Engineering")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_comparison(comparisons: list[RegressionComparison]) -> None:
    model_names: list[str] = [c.model_name for c in comparisons]
    baseline_r2: list[float] = [c.baseline_test_r2 for c in comparisons]
    engineered_r2: list[float] = [c.engineered_test_r2 for c in comparisons]

    x = np.arange(len(model_names))
    width: float = 0.35

    _fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline_r2, width, label="Baseline", color="steelblue")
    ax.bar(x + width / 2, engineered_r2, width, label="With Engineered Features", color="tomato")

    ax.set_title("Regression Test R² — Before vs After Feature Engineering")
    ax.set_ylabel("Test R²")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    df_baseline = get_clean_df()
    df_engineered = engineer(df_baseline)

    ridge_artifacts = get_ridge_artifacts()
    lasso_artifacts = get_lasso_artifacts()

    console.print(
        Panel(
            f"[bold]Baseline dataset:[/bold] {df_baseline.shape[0]} rows, {df_baseline.shape[1]} columns\n"
            f"[bold]Engineered dataset:[/bold] {df_engineered.shape[0]} rows, {df_engineered.shape[1]} columns\n"
            f"[bold]New features:[/bold] {', '.join(ENGINEERED_FEATURES)}",
            title="Workforce Attrition — Bonus 3: Feature Engineering and Model Improvement",
        ),
    )

    report_engineered_features()

    tree_comparison = compare_trees(df_baseline, df_engineered)
    regression_comparisons = compare_regression(df_baseline, df_engineered, ridge_artifacts, lasso_artifacts)

    report_tree_comparison(tree_comparison)
    report_regression_comparison(regression_comparisons)

    plot_correlation(df_engineered)
    plot_tree_comparison(tree_comparison)
    plot_regression_comparison(regression_comparisons)


if __name__ == "__main__":
    main()
