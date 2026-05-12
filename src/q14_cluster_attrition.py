import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from rich.panel import Panel
from rich.table import Table

from q01_data_quality import console
from q12_linkage_comparison import COMPLETE_CLUSTER_PATH, CompleteLinkageArtifacts
from q13_cluster_profiling import CLUSTER_NAMES

ATTRITION_LABELS: dict[int, str] = {0: "Stayed", 1: "Left"}


def load_artifacts() -> CompleteLinkageArtifacts:
    return CompleteLinkageArtifacts(**joblib.load(COMPLETE_CLUSTER_PATH))


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
    table.add_column("Cluster", style="cyan")
    table.add_column("Stayed", justify="right")
    table.add_column("Left", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Attrition Rate %", justify="right")

    for cluster_name, row in crosstab.iterrows():
        rate: float = row["Attrition_Rate_%"]
        color = "red" if rate > 60 else "yellow" if rate > 40 else "green"
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
    console.print(f"\n[bold]Highest attrition cluster:[/bold] [red]{highest_cluster}[/red] ({highest_rate:.2f}%)")
    console.print(
        "[dim]Clustering reveals hidden employee segments that attrition labels alone cannot surface. "
        "The Flight Risk cluster concentrates low performers with low satisfaction — a pattern "
        "invisible in aggregate attrition statistics.[/dim]",
    )


def report_clustering_value() -> None:
    table = Table(title="How Clustering Supports Classification", show_lines=True)
    table.add_column("Aspect", style="cyan")
    table.add_column("Insight")

    table.add_row(
        "Hidden groups",
        "Clustering surfaces behavioral segments without using Attrition as input",
    )
    table.add_row(
        "Feature enrichment",
        "Cluster membership can be added as a feature to the Decision Tree to improve predictions",
    )
    table.add_row(
        "Targeted intervention",
        "Unlike binary attrition labels, clusters explain WHY employees are at risk",
    )
    table.add_row(
        "Validation",
        "High attrition concentration in Flight Risk cluster confirms cluster validity",
    )

    console.print(table)


def plot_attrition_heatmap(frame: pd.DataFrame) -> None:
    crosstab = pd.crosstab(frame["Cluster_Name"], frame["Attrition_Label"])

    _fig, ax = plt.subplots(figsize=(7, 4))
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
    rate_df = frame.groupby("Cluster_Name")["Attrition"].mean().mul(100).reset_index().rename(columns={"Attrition": "Attrition_Rate_%"}).sort_values("Attrition_Rate_%", ascending=False)

    _fig, ax = plt.subplots(figsize=(8, 5))
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
    artifacts = load_artifacts()
    frame = build_analysis_frame(artifacts)

    console.print(
        Panel(
            f"[bold]Sample size:[/bold] {len(frame)}  [bold]Clusters:[/bold] {frame['Cluster'].nunique()}  [bold]Attrition rate:[/bold] {frame['Attrition'].mean() * 100:.1f}%",
            title="Workforce Attrition — Q14 Cluster and Attrition Analysis",
        ),
    )

    report_attrition_by_cluster(frame)
    report_clustering_value()
    plot_attrition_heatmap(frame)
    plot_attrition_rate_by_cluster(frame)


if __name__ == "__main__":
    main()
