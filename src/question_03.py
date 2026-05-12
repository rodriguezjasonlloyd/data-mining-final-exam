"""Q3. Correlation Analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table

from question_01 import clean, console, load_raw

EDUCATION_ORDER = ["High School", "Vocational", "Bachelor", "Master", "Phd"]

SALARY_PREDICTORS = [
    "Monthly_Salary_PHP",
    "Tenure_Years",
    "Education_Level_Encoded",
    "Performance_Score",
    "Num_Promotions",
]


def encode_education(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Education_Level_Encoded"] = result["Education_Level"].str.title().map({level: rank for rank, level in enumerate(EDUCATION_ORDER)})
    return result


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[SALARY_PREDICTORS].corr(method="pearson")


def report_salary_correlations(correlation: pd.DataFrame) -> None:
    salary_correlation = correlation["Monthly_Salary_PHP"].drop("Monthly_Salary_PHP").sort_values(key=abs, ascending=False)

    table = Table(title="Pearson Correlation with Monthly_Salary_PHP", show_lines=True)
    table.add_column("Predictor", style="cyan")
    table.add_column("Correlation", justify="right")
    table.add_column("Strength", justify="right")

    for predictor, value in salary_correlation.items():
        absolute_value = abs(value)
        strength = "strong" if absolute_value > 0.5 else "moderate" if absolute_value > 0.3 else "weak"
        color = "green" if absolute_value > 0.5 else "yellow" if absolute_value > 0.3 else "dim"
        table.add_row(str(predictor), f"{value:.4f}", f"[{color}]{strength}[/{color}]")

    console.print(table)

    strongest = salary_correlation.index[0]
    console.print(f"\n[bold]Strongest predictor:[/bold] {strongest} (r={salary_correlation[strongest]:.4f})")


def report_multicollinearity(correlation: pd.DataFrame) -> None:
    table = Table(title="Predictor Intercorrelations (multicollinearity check)", show_lines=True)
    predictors = [col for col in SALARY_PREDICTORS if col != "Monthly_Salary_PHP"]

    table.add_column("Pair", style="cyan")
    table.add_column("Correlation", justify="right")
    table.add_column("Risk", justify="right")

    for index, column_a in enumerate(predictors):
        for column_b in predictors[index + 1 :]:
            value = correlation.loc[column_a, column_b]
            absolute_value = abs(value)
            risk = "high" if absolute_value > 0.7 else "moderate" if absolute_value > 0.5 else "low"
            color = "red" if absolute_value > 0.7 else "yellow" if absolute_value > 0.5 else "green"
            table.add_row(f"{column_a} x {column_b}", f"{value:.4f}", f"[{color}]{risk}[/{color}]")

    console.print(table)


def plot_heatmap(corr: pd.DataFrame) -> None:
    _fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap — Salary Predictors")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = encode_education(clean(load_raw()))

    console.print(
        Panel(
            f"[bold]Cleaned dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns",
            title="Workforce Attrition — Q03 Correlation Analysis",
        ),
    )

    correlations = compute_correlations(df)

    report_salary_correlations(correlations)
    report_multicollinearity(correlations)
    plot_heatmap(correlations)


if __name__ == "__main__":
    main()
