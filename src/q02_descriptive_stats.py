import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table

from q01_data_quality import clean, console, load_raw

NUMERIC_COLUMNS = [
    "Age",
    "Monthly_Salary_PHP",
    "Tenure_Years",
    "Performance_Score",
    "Training_Hours_YTD",
    "Absences_YTD",
    "Overtime_Hours_Monthly",
    "Distance_Office_KM",
    "Job_Satisfaction_Score",
    "Work_Life_Balance_Score",
    "Num_Promotions",
    "Prev_Companies",
]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_COLUMNS].describe().T


def compute_skewness(df: pd.DataFrame) -> pd.Series:
    return df[NUMERIC_COLUMNS].skew().sort_values(key=abs, ascending=False)


def report_summary(summary: pd.DataFrame) -> None:
    table = Table(title="Descriptive Statistics", show_lines=True)
    table.add_column("Variable", style="cyan")
    for col in summary.columns:
        table.add_column(col, justify="right")

    for variable, row in summary.iterrows():
        table.add_row(str(variable), *[f"{v:.2f}" for v in row])

    console.print(table)


def report_skewness(skewness: pd.Series) -> None:
    table = Table(title="Skewness — All Numeric Variables", show_lines=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Skewness", justify="right")
    table.add_column("Direction", justify="right")

    for variable, value in skewness.items():
        direction = "[red]positive[/red]" if value > 0 else "[blue]negative[/blue]"
        highlight = "[bold]" if variable in skewness.head(3).index else ""
        table.add_row(f"{highlight}{variable}", f"{highlight}{value:.4f}", direction)

    console.print(table)


def plot_top_skewed(df: pd.DataFrame, skewness: pd.Series) -> None:
    top3 = skewness.head(3).index.tolist()
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, column in zip(axes, top3, strict=False):
        skew_val = skewness[column]
        sns.histplot(df[column], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"{column}\nskew={skew_val:.4f}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    plt.suptitle("Top 3 Most Skewed Variables")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = clean(load_raw())

    console.print(
        Panel(
            f"[bold]Cleaned dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns",
            title="Workforce Attrition — Q02 Descriptive Statistics",
        ),
    )

    summary = compute_summary(df)
    skewness = compute_skewness(df)

    report_summary(summary)
    report_skewness(skewness)
    plot_top_skewed(df, skewness)


if __name__ == "__main__":
    main()
