"""Q1. Data Quality Assessment."""

from functools import cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _init_backend() -> str:
    for backend in ("TkAgg", "Qt5Agg", "MacOSX", "Agg"):
        try:
            mpl.use(backend)
            plt.figure()
            plt.close()
        except (ImportError, OSError) as error:
            console.print(f"[dim]Backend {backend} unavailable: {error}[/dim]")
            continue
        else:
            return backend
    return "Agg"


console = Console()

_init_backend()


@cache
def get_clean_df() -> pd.DataFrame:
    return clean(load_raw())


AGE_MIN = 15
AGE_MAX = 80
SALARY_MAX = 500_000

DEPARTMENT_MAP: dict[str, str] = {
    "Operations": "Operations",
    "OPERATIONS": "Operations",
    "operations": "Operations",
    "Operationsn": "Operations",
    "Ops": "Operations",
    "Sales": "Sales",
    "SALES": "Sales",
    "sales": "Sales",
    "Saless": "Sales",
    "Finance": "Finance",
    "FINANCE": "Finance",
    "finance": "Finance",
    "Fiance": "Finance",
    "IT": "IT",
    "it": "IT",
    "I.T.": "IT",
    "Information Technology": "IT",
    "HR": "HR",
    "hr": "HR",
    "H.R.": "HR",
    "Human Resources": "HR",
    "Customer Service": "Customer Service",
    "Marketing": "Marketing",
    "Compliance": "Compliance",
}


def load_raw() -> pd.DataFrame:
    return pd.read_csv("data.csv")


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query("missing_count > 0").sort_values("missing_pct", ascending=False)


def check_suspicious_numerics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "age": df[df["Age"].notna() & ((df["Age"] < AGE_MIN) | (df["Age"] > AGE_MAX))][["Employee_ID", "Age"]],
        "salary_low": df[df["Monthly_Salary_PHP"].notna() & (df["Monthly_Salary_PHP"] <= 0)][["Employee_ID", "Monthly_Salary_PHP"]],
        "salary_high": df[df["Monthly_Salary_PHP"].notna() & (df["Monthly_Salary_PHP"] > SALARY_MAX)][["Employee_ID", "Monthly_Salary_PHP"]],
        "absences": df[df["Absences_YTD"].notna() & (df["Absences_YTD"] < 0)][["Employee_ID", "Absences_YTD"]],
        "tenure": df[df["Tenure_Years"].notna() & (df["Tenure_Years"] < 0)][["Employee_ID", "Tenure_Years"]],
        "performance": df[df["Performance_Score"].notna() & ~df["Performance_Score"].between(1, 5)][["Employee_ID", "Performance_Score"]],
        "satisfaction": df[df["Job_Satisfaction_Score"].notna() & ~df["Job_Satisfaction_Score"].between(1, 10)][["Employee_ID", "Job_Satisfaction_Score"]],
    }


def check_department_names(df: pd.DataFrame) -> pd.Series:
    return df["Department"].value_counts(dropna=False)


def check_categorical_values(df: pd.DataFrame) -> dict[str, pd.Series]:
    cats = ["Gender", "Marital_Status", "Education_Level", "Employment_Type", "Shift", "Performance_Rating"]
    return {column: df[column].value_counts(dropna=False) for column in cats if column in df.columns}


def get_dropped_rows(df: pd.DataFrame) -> pd.DataFrame:
    age_mask = df["Age"].notna() & ((df["Age"] < AGE_MIN) | (df["Age"] > AGE_MAX))
    salary_low_mask = df["Monthly_Salary_PHP"].notna() & (df["Monthly_Salary_PHP"] <= 0)
    salary_high_mask = df["Monthly_Salary_PHP"].notna() & (df["Monthly_Salary_PHP"] > SALARY_MAX)
    return df[age_mask | salary_low_mask | salary_high_mask][["Employee_ID", "Age", "Monthly_Salary_PHP"]].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Department"] = result["Department"].map(DEPARTMENT_MAP).fillna(result["Department"])

    for column in ["Gender", "Marital_Status", "Education_Level", "Employment_Type", "Shift", "Region", "Performance_Rating"]:
        if column in result.columns:
            result[column] = result[column].str.strip().str.title()

    result = result[~(result["Age"].notna() & ((result["Age"] < AGE_MIN) | (result["Age"] > AGE_MAX)))]
    result = result[~(result["Monthly_Salary_PHP"].notna() & (result["Monthly_Salary_PHP"] <= 0))]
    result = result[~(result["Monthly_Salary_PHP"].notna() & (result["Monthly_Salary_PHP"] > SALARY_MAX))]

    result["Absences_YTD"] = result["Absences_YTD"].clip(lower=0)
    result["Tenure_Years"] = result["Tenure_Years"].clip(lower=0)

    numeric_columns = [
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
    for column in numeric_columns:
        if column in result.columns:
            result[column] = result[column].fillna(result[column].median())

    categorical_columns = [
        "Gender",
        "Marital_Status",
        "Region",
        "Education_Level",
        "Department",
        "Employment_Type",
        "Shift",
        "Performance_Rating",
    ]
    for column in categorical_columns:
        if column in result.columns:
            result[column] = result[column].fillna(result[column].mode()[0])

    return result


def report_missing(missing_df: pd.DataFrame) -> None:
    table = Table(title="Missing Values", show_lines=True)
    table.add_column("Column", style="cyan")
    table.add_column("Missing Count", justify="right")
    table.add_column("Missing %", justify="right")

    for column, row in missing_df.iterrows():
        pct = row["missing_pct"]
        color = "red" if pct > 5 else "yellow" if pct > 2 else "green"
        table.add_row(str(column), str(int(row["missing_count"])), f"[{color}]{pct:.2f}%[/{color}]")

    console.print(table)


def report_suspicious(suspicious: dict[str, pd.DataFrame]) -> None:
    table = Table(title="Suspicious Values", show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Decision", style="dim")

    decisions = {
        "age": f"Drop rows (Age < {AGE_MIN} or > {AGE_MAX})",
        "salary_low": "Drop rows (Salary ≤ 0)",
        "salary_high": f"Drop rows (Salary > {SALARY_MAX:,})",
        "absences": "Clip to 0 (negative absences)",
        "tenure": "Clip to 0 (negative tenure)",
        "performance": "No issues found",
        "satisfaction": "No issues found",
    }

    for key, flagged in suspicious.items():
        count = len(flagged)
        color = "red" if count > 0 else "green"
        table.add_row(key, f"[{color}]{count}[/{color}]", decisions[key])

    console.print(table)


def report_dropped(dropped: pd.DataFrame) -> None:
    if dropped.empty:
        console.print("[green]No rows dropped.[/green]")
        return

    table = Table(title=f"Rows to be Dropped ({len(dropped)} total)", show_lines=True)
    table.add_column("Employee_ID", style="cyan")
    table.add_column("Age", justify="right")
    table.add_column("Monthly_Salary_PHP", justify="right")
    table.add_column("Reason", style="red")

    for _, row in dropped.iterrows():
        reasons = []
        age = row["Age"]
        salary = row["Monthly_Salary_PHP"]
        if pd.notna(age) and (age < AGE_MIN or age > AGE_MAX):
            reasons.append(f"Age {age} out of [{AGE_MIN}, {AGE_MAX}]")
        if pd.notna(salary) and salary <= 0:
            reasons.append(f"Salary {salary} ≤ 0")
        if pd.notna(salary) and salary > SALARY_MAX:
            reasons.append(f"Salary {salary:,} > {SALARY_MAX:,}")
        table.add_row(str(row["Employee_ID"]), str(age), str(salary), "; ".join(reasons))

    console.print(table)


def report_departments(departments: pd.Series) -> None:
    unmapped = [str(name) for name in departments.index if str(name) not in DEPARTMENT_MAP]
    status = "Fully Mapped" if not unmapped else f"Unmapped: {', '.join(unmapped)}"
    table = Table(title=f"Department Name Variants ({status})", show_lines=True)
    table.add_column("Raw Value", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Maps To", style="green")

    for name, count in departments.items():
        canonical = DEPARTMENT_MAP.get(str(name), "[red]UNMAPPED[/red]")
        table.add_row(str(name), str(count), canonical)

    console.print(table)


def report_categoricals(cats: dict[str, pd.Series]) -> None:
    for column, counts in cats.items():
        table = Table(title=f"{column} Distribution", show_lines=True)
        table.add_column("Value", style="cyan")
        table.add_column("Count", justify="right")
        for val, count in counts.items():
            table.add_row(str(val), str(count))
        console.print(table)


def plot_missing(missing_df: pd.DataFrame) -> None:
    if missing_df.empty:
        return

    _fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(data=missing_df.reset_index(), x="index", y="missing_pct", ax=ax, color="steelblue")
    ax.set_title("Missing Values by Column (%)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing %")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    columns = ["Age", "Monthly_Salary_PHP", "Absences_YTD", "Tenure_Years"]
    _fig, axes = plt.subplots(1, len(columns), figsize=(16, 4))

    for ax, column in zip(axes, columns, strict=False):
        if column in df.columns:
            sns.boxplot(y=df[column], ax=ax, color="steelblue")
            ax.set_title(column)

    plt.suptitle("Numeric Distributions — Pre-cleaning")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_raw()

    console.print(
        Panel(
            f"[bold]Raw dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns",
            title="Workforce Attrition — Q01 Data Quality",
        ),
    )

    missing_df = check_missing(df)
    suspicious = check_suspicious_numerics(df)
    dept = check_department_names(df)
    cats = check_categorical_values(df)
    dropped = get_dropped_rows(df)

    report_missing(missing_df)
    report_suspicious(suspicious)
    report_dropped(dropped)
    report_departments(dept)
    report_categoricals(cats)

    plot_numeric_distributions(df)
    plot_missing(missing_df)

    clean_df = get_clean_df()

    console.print(
        Panel(
            f"[green]Cleaned dataset:[/green] {clean_df.shape[0]} rows x {clean_df.shape[1]} columns\n"
            f"[red]Rows removed:[/red] {df.shape[0] - clean_df.shape[0]} (impossible age/salary values)\n"
            f"[yellow]Age threshold:[/yellow] [{AGE_MIN}, {AGE_MAX}]\n"
            f"[yellow]Salary threshold:[/yellow] (0, {SALARY_MAX:,}]",
            title="Cleaning Summary",
        ),
    )


if __name__ == "__main__":
    main()
