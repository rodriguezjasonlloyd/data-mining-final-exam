"""Q14. Cluster and Attrition Analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from rich.table import Table

from question_01 import console
from question_12 import CompleteLinkageArtifacts, get_complete_cluster_artifacts
from question_13 import CLUSTER_NAMES

ATTRITION_LABELS: dict[int, str] = {0: "Stayed", 1: "Left"}


def build_analysis_frame(artifacts: CompleteLinkageArtifacts) -> pd.DataFrame:
    frame = artifacts.sample_df.copy()
    frame["Cluster"] = artifacts.cluster_labels
    frame["Cluster_Name"] = frame["Cluster"].map(CLUSTER_NAMES)
    frame["Attrition_Label"] = frame["Attrition"].map(ATTRITION_LABELS)
    return frame


def report_attrition_by_cluster(frame: pd.DataFrame) -> None:
    crosstab = pd.crosstab(frame["Cluster_Name"], frame["Attrition_Label"])
    crosstab["Total"] = crosstab.sum(axis=1)
    crosstab["Attrition_Rate_%"] = (crosstab.get("Left", 0) / crosstab["Total"] * 100).round(2)
    crosstab = crosstab.sort_values("Attrition_Rate_%", ascending=False)

    table = Table(title="Attrition Distribution by Cluster", show_lines=True)
    table.add_column("Cluster", style="bright_cyan")
    table.add_column("Stayed", justify="right")
    table.add_column("Left", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Attrition Rate %", justify="right")

    for cluster_name, row in crosstab.iterrows():
        rate: float = row["Attrition_Rate_%"]
        color = "bright_red" if rate > 60 else "bright_yellow" if rate > 40 else "bright_green"
        table.add_row(
            str(cluster_name),
            str(int(row.get("Stayed", 0))),
            str(int(row.get("Left", 0))),
            str(int(row["Total"])),
            f"[{color}]{rate:.2f}%[/{color}]",
        )

    console.print(table)

    highest_cluster: str = str(crosstab.index[0])
    highest_rate: float = float(crosstab.iloc[0]["Attrition_Rate_%"])
    console.print(f"\n[bold]Highest attrition cluster:[/bold] [bright_red]{highest_cluster}[/bright_red] ({highest_rate:.2f}%)")


def plot_attrition_heatmap(frame: pd.DataFrame) -> None:
    crosstab = pd.crosstab(frame["Cluster_Name"], frame["Attrition_Label"])

    _, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Attrition Count by Cluster")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_attrition_rate_by_cluster(frame: pd.DataFrame) -> None:
    rate_df = (
        frame.groupby("Cluster_Name")["Attrition"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"Attrition": "Attrition_Rate_%"})
        .sort_values("Attrition_Rate_%", ascending=False)
    )

    _, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=rate_df,
        x="Cluster_Name",
        y="Attrition_Rate_%",
        hue="Cluster_Name",
        palette={"Stable Performers": "steelblue", "Outlier Pair": "goldenrod", "Flight Risk": "tomato"},
        legend=False,
        ax=ax,
    )

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="{:.1f}%", padding=3, fontsize=10)

    ax.set_title("Attrition Rate by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Attrition Rate (%)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = get_complete_cluster_artifacts()
    frame = build_analysis_frame(artifacts)

    console.print(f"[bold]Attrition rate:[/bold] {frame['Attrition'].mean() * 100:.1f}%\n")

    report_attrition_by_cluster(frame)
    plot_attrition_heatmap(frame)
    plot_attrition_rate_by_cluster(frame)


if __name__ == "__main__":
    main()
