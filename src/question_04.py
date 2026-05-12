from question_01 import console, clean, load_raw
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table


def compute_attrition_rate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    crosstab = pd.crosstab(df[by], df["Attrition"], normalize="index") * 100
    crosstab.columns = ["Retained_%", "Attrition_%"]
    crosstab["Total"] = df.groupby(by)["Attrition"].count()
    return crosstab.sort_values("Attrition_%", ascending=False).round(2)


def report_attrition(rates: pd.DataFrame, by: str) -> None:
    table = Table(title=f"Attrition Rate by {by}", show_lines=True)
    table.add_column(by, style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Attrition %", justify="right")
    table.add_column("Retained %", justify="right")

    for group, row in rates.iterrows():
        pct = row["Attrition_%"]
        color = "red" if pct > 30 else "yellow" if pct > 15 else "green"
        table.add_row(str(group), str(int(row["Total"])), f"[{color}]{pct:.2f}%[/{color}]", f"{row['Retained_%']:.2f}%")

    highest = rates.index[0]
    lowest = rates.index[-1]
    console.print(table)
    console.print(f"  [red]Highest:[/red] {highest} ({rates.loc[highest, 'Attrition_%']:.2f}%)  [green]Lowest:[/green] {lowest} ({rates.loc[lowest, 'Attrition_%']:.2f}%)\n")


def plot_attrition(df: pd.DataFrame, by: str) -> None:
    plot_df = df.copy()
    plot_df["Attrition_Label"] = plot_df["Attrition"].map({0: "Stayed", 1: "Left"})

    _fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=plot_df, x=by, hue="Attrition_Label", ax=ax)
    ax.set_title(f"Attrition Count by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = clean(load_raw())

    console.print(
        Panel(
            f"[bold]Cleaned dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns",
            title="Workforce Attrition — Q04 Attrition Frequency Analysis",
        ),
    )

    for by in ["Department", "Employment_Type"]:
        rates = compute_attrition_rate(df, by)
        report_attrition(rates, by)
        plot_attrition(df, by)


if __name__ == "__main__":
    main()
